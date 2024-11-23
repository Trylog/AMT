import os
import tensorflow as tf
import numpy as np
import librosa
import pretty_midi
import keras
from keras import layers, models, mixed_precision
from keras.src.applications.efficientnet import EfficientNetB0
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
                yield np.array(X), np.array(y)





#model
def build_model(input_shape, num_notes=88, timesteps=224):
    # Ekstraktor cech (EfficientNet)
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Zamrożenie wag pretrenowanego modelu

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)  # Wyjście np. (7, 7, 1280)
    x = layers.Reshape((timesteps, -1))(x)  # Dopasowanie do wymiaru czasowego (224, 1280)

    # Warstwy LSTM dla modelowania zależności czasowych
    x = layers.LSTM(256, return_sequences=True)(x)  # (224, 256)
    x = layers.LSTM(128, return_sequences=True)(x)  # (224, 128)

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
batch_size = 8
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

#kompilacja modelu
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

#trenowanie modelu z checkpointami
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=[checkpoint]
)

#zapisanie finalnego modelu
model.save("final_model.keras")

#ewaluacja modelu
test_generator = data_generator(audio_test, midi_test, batch_size)
test_steps = len(audio_test) // batch_size
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
