import os
import time

import tensorflow as tf
import numpy as np
import librosa
import pretty_midi
import keras
from keras import layers, models, mixed_precision, callbacks
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.backend.tensorflow.nn import binary_crossentropy
from sklearn.model_selection import train_test_split
import soundfile as sf


#sprawdzenie dostępności GPU
#print("Czy GPU jest dostępne?", tf.config.list_physical_devices('GPU'))
#ustawienie mieszanej precyzji
mixed_precision.set_global_policy('mixed_float16')

#generowanie log-mel spektrogramu
def generate_spectrogram(audio_path, sr=44100, n_mels=224, hop_length=512):
    y, sr = librosa.load(audio_path)
    print(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

#konwersja MIDI na macierz aktywacji
def midi_to_activation_matrix(midi_path, sr=22050, hop_length=512, num_notes=88):
    midi = pretty_midi.PrettyMIDI(midi_path)
    max_time = midi.get_end_time()
    frames = int(np.ceil(max_time * sr / hop_length))
    activation_matrix = np.zeros((num_notes, frames), dtype=np.float32)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start_frame = int(np.floor(note.start * sr / hop_length))
            end_frame = int(np.ceil(note.end * sr / hop_length))
            if 21 <= note.pitch <= 108:
                activation_matrix[note.pitch - 21, start_frame:end_frame] = 1.0

    return activation_matrix

#przygotowanie danych: dzielenie spektrogramów na segmenty czasowe
def prepare_dataset(audio_paths, midi_paths, sr=22050, hop_length=512):
    for audio_path, midi_path in zip(audio_paths, midi_paths):
        spectrogram = generate_spectrogram(audio_path, sr=sr, hop_length=hop_length)
        activation_matrix = midi_to_activation_matrix(midi_path, sr=sr, hop_length=hop_length)
        min_length = min(spectrogram.shape[1], activation_matrix.shape[0])
        spectrogram = spectrogram[:, :min_length]
        activation_matrix = activation_matrix[:min_length, :]
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        yield spectrogram, activation_matrix

#generator danych
def data_generator(audio_paths, midi_paths, batch_size, sr=22050, hop_length=512):
    while True:
        for i in range(0, len(audio_paths), batch_size):
            batch_audio_paths = audio_paths[i:i + batch_size]
            batch_midi_paths = midi_paths[i:i + batch_size]

            X = []
            y = []

            for audio_path, midi_path in zip(batch_audio_paths, batch_midi_paths):

                spectrogram = generate_spectrogram(audio_path, sr=sr, hop_length=hop_length)
                spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, axis=-1)
                activation_matrix = midi_to_activation_matrix(midi_path, sr=sr, hop_length=hop_length)
                #print(spectrogram.shape)

                number_of_model_frames = min(int(spectrogram.shape[1] / 224), int(activation_matrix.shape[1] / 224))

                for j in range(number_of_model_frames):
                    X.append(spectrogram[:,j * 224:(j + 1) * 224,:])
                    y.append(activation_matrix[:,j * 224:(j + 1) * 224])
                print(np.array(X).shape)
                print(np.array(y).shape)
                print(i)
                yield np.array(X), np.array(y)


#model
def build_model(input_shape, num_notes=88, timesteps=224):
    # Ekstraktor cech (EfficientNet)
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-20]:  # Zamrażamy wszystkie warstwy poza ostatnimi 20
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)  # Wyjście np. (7, 7, 1280)
    x = layers.Reshape((timesteps, -1))(x)  # Dopasowanie do wymiaru czasowego (224, 1280)

    # Warstwy LSTM dla modelowania zależności czasowych
    x = layers.LSTM(256, return_sequences=True, dropout=0.3)(x)  # (224, 256)
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)  # (224, 128)

    # Wyjście dopasowane do MIDI
    x = layers.TimeDistributed(layers.Dense(num_notes, activation="sigmoid"))(x)  # (224, 88)
    x = layers.Permute((2, 1))(x) # (88, 224)

    model = models.Model(inputs, x)
    return model

#ścieżki do plików audio i MIDI
audio_paths = []
midi_paths = []

for file in os.listdir('train_model\wav\Mozart'):
    audio_paths.append(os.path.join('train_model\wav\Mozart', file))
    midi_paths.append(os.path.join('train_model\midi\Mozart', file[:-3] + 'midi'))
for file in os.listdir('train_model\wav\Beethoven'):
    audio_paths.append(os.path.join('train_model\wav\Beethoven', file))
    midi_paths.append(os.path.join('train_model\midi\Beethoven', file[:-3] + 'midi'))
for file in os.listdir('train_model\wav\Haydn'):
    audio_paths.append(os.path.join('train_model\wav\Haydn', file))
    midi_paths.append(os.path.join('train_model\midi\Haydn', file[:-3] + 'midi'))

#podział na treningowy (70%), walidacyjny (15%) i testowy (15%)
audio_train, audio_temp, midi_train, midi_temp = train_test_split(
    audio_paths, midi_paths, test_size=0.3, random_state=42
)
audio_val, audio_test, midi_val, midi_test = train_test_split(
    audio_temp, midi_temp, test_size=0.5, random_state=42
)

print("Treningowy:", len(audio_train))
print("Walidacyjny:", len(audio_val))
print("Testowy:", len(audio_test))

#parametry generatora i modelu
batch_size = 4
input_shape = (224, 224, 3)
train_generator = data_generator(audio_train, midi_train, batch_size)
val_generator = data_generator(audio_val, midi_val, batch_size)
steps_per_epoch = len(audio_train) // batch_size
validation_steps = len(audio_val) // batch_size

#budowanie modelu
model = build_model(input_shape)

#checkpointy do zapisywania modelu po każdej epoce
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras")

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0004,
    decay_steps=200,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

print(model.summary())

#zmniejsza szybkość uczenia przy małej mianie wyjścia
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#trenowanie modelu z checkpointami
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=[checkpoint, reduce_lr, early_stopping]
)

#zapisanie finalnego modelu
model.save("final_model_new_new.keras")

from sklearn.metrics import precision_score, recall_score, f1_score


# Funkcja pomocnicza dla ewaluacji na zbiorze testowym
def evaluate_model(model, generator, steps):
    all_y_true = []
    all_y_pred = []

    for i, (X_batch, y_batch) in enumerate(generator):
        if i >= steps:  # Ograniczenie do liczby kroków w teście
            break
        y_pred = model.predict(X_batch)  # Predykcje modelu
        y_pred = (y_pred > 0.5).astype(int)  # Binarne progowanie
        all_y_true.append(y_batch)
        all_y_pred.append(y_pred)

    # Konwersja na macierze
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    # Obliczenie metryk
    precision = precision_score(y_true.flatten(), y_pred.flatten(), average="micro")
    recall = recall_score(y_true.flatten(), y_pred.flatten(), average="micro")
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average="micro")

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    return precision, recall, f1


#ewaluacja modelu
print(time.localtime())
test_generator = data_generator(audio_test, midi_test, batch_size)
test_steps = len(audio_test) // batch_size
precision, recall, f1 = evaluate_model(model, test_generator, test_steps)
print(f"Test precision: {precision}, Test recall: {recall}, Test f1: {f1}")
