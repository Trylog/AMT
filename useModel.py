import keras
import librosa
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, sr=44100, n_mels=224, hop_length=512):
    y, sr = librosa.load(audio_path)
    print(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram, sr

model = keras.models.load_model("final_model.keras")

# Generowanie spektrogramu dla nowego pliku audio
audio_path = "train_model/wav/Mozart/MIDI-Unprocessed_XP_14_R1_2004_04_ORIG_MID--AUDIO_14_R1_2004_06_Track06_wav.wav"
spectrogram, sr = generate_spectrogram(audio_path)

# fig, ax = plt.subplots()
#
# img = librosa.display.specshow(spectrogram, x_axis='time',
#                          y_axis='mel', sr=sr,
#                          fmax=8000, ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
#plt.show()

spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, axis=-1)  # Rozszerzenie osi

# Dzielenie na segmenty (timesteps = 224)
timesteps = 224
frames = spectrogram.shape[1] // timesteps
input_data = [
    spectrogram[:, i * timesteps:(i + 1) * timesteps, :]
    for i in range(frames)
]
input_data = np.array(input_data)

# Predykcja
predictions = model.predict(input_data)  # (n_frames, 88, 224)
activation_matrix = np.hstack(predictions)  # Łączenie segmentów w jedną macierz


def activation_matrix_to_midi(activation_matrix, sr=22050, hop_length=512, threshold=0.5):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Domyślny instrument: fortepian

    num_notes, num_frames = activation_matrix.shape
    time_per_frame = hop_length / sr

    for note_idx in range(num_notes):
        note_active = False
        start_time = 0

        for frame_idx in range(num_frames):
            if activation_matrix[note_idx, frame_idx] >= threshold and not note_active:
                # Start nowej nuty
                note_active = True
                start_time = frame_idx * time_per_frame

            if activation_matrix[note_idx, frame_idx] < threshold and note_active:
                # Zakończenie nuty
                note_active = False
                end_time = frame_idx * time_per_frame
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_idx + 21,  # MIDI pitch: 21 = A0
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

        # Jeśli nuta aktywna do końca, zamknij ją
        if note_active:
            end_time = num_frames * time_per_frame
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_idx + 21,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


# Konwersja
midi = activation_matrix_to_midi(activation_matrix)
midi.write("output.mid")
