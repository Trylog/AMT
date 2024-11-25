import os
import numpy as np
import librosa
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboard import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from sklearn.model_selection import train_test_split


# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Czy GPU jest dostępne?", device)


# Generowanie log-mel spektrogramu
def generate_spectrogram(audio_path, sr=44100, n_mels=224, hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


# Konwersja MIDI na macierz aktywacji
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


#Dataset dla PyTorcha
class AMTDataset(Dataset):
    def __init__(self, audio_paths, midi_paths, sr=22050, hop_length=512, batch_size=2):
        self.audio_paths = audio_paths
        self.midi_paths = midi_paths
        self.sr = sr
        self.hop_length = hop_length
        self.batch_size = batch_size #długość jednego zestawu ramek 224x224, czym większa tym większy kontekst
        self.total_segments = self.calculate_total_segments()

        self.current_audio_idx = -1  # Indeks aktualnie załadowanego spektrogramu
        self.spectrogram = None  # Aktualnie przetwarzany spektrogram
        self.activation_matrix = None  # Macierz aktywacji
        self.current_segment_idx = 0  # Aktualny indeks segmentu

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index):
        # Załaduj kolejny spektrogram, jeśli segmenty się skończyły
        while self.spectrogram is None or self.current_segment_idx >= self.spectrogram.shape[1] // 224 // self.batch_size:
            self.load_next_spectrogram()

        spectrogram_segments = []
        activation_segments = []
        print("starting segment loading")
        # Wyciągnij segment
        for i in range(self.batch_size):
            start_frame = (self.current_segment_idx * self.batch_size + i) * 224
            end_frame = start_frame + 224

            spectrogram_segments.append(self.spectrogram[:, start_frame:end_frame, :])
            activation_segments.append(self.activation_matrix[:, start_frame:end_frame])
            print("frame")

        self.current_segment_idx += 1  # Przejdź do następnego segmentu

        spectrogram_segments = [np.transpose(segment, (2, 0, 1)) for segment in spectrogram_segments]
        print("batch finished: " + str(self.current_audio_idx) + ": " + str(np.array(spectrogram_segments).shape))

        return torch.tensor(np.array(spectrogram_segments), dtype=torch.float32), torch.tensor(np.array(activation_segments), dtype=torch.float32)

    def calculate_total_segments(self):
        #Oblicza całkowitą liczbę segmentów w dataset.
        total_segments = 0
        for audio_path, midi_path in zip(self.audio_paths, self.midi_paths):
            spectrogram_samples = self.get_spectrogram_shape(audio_path)
            activation_samples = self.get_activation_matrix_shape(midi_path)
            print(audio_path)
            num_segments = min(
                spectrogram_samples // 224 // self.batch_size,
                activation_samples // 224 // self.batch_size,
            )
            total_segments += num_segments
        print("calculated total segments", total_segments)
        return total_segments

    def get_spectrogram_shape(self, audio_path):
        #Zwraca liczbę sampli w spektrogramie bez jego pełnej generacji.
        y, _ = librosa.load(audio_path, sr=self.sr)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=224, hop_length=self.hop_length)
        return spectrogram.shape[1]  # Liczba sampli

    def get_activation_matrix_shape(self, midi_path):
        #Zwraca liczbę ramek czasowych w macierzy aktywacji.
        midi = pretty_midi.PrettyMIDI(midi_path)
        max_time = midi.get_end_time()
        frames = int(np.ceil(max_time * self.sr / self.hop_length))
        return frames

    def load_next_spectrogram(self):
        #Ładuje następny spektrogram i odpowiadającą mu macierz aktywacji.
        self.current_audio_idx += 1

        if self.current_audio_idx >= len(self.audio_paths):
            raise StopIteration("All audio files processed.")

        audio_path = self.audio_paths[self.current_audio_idx]
        midi_path = self.midi_paths[self.current_audio_idx]

        self.spectrogram = generate_spectrogram(audio_path)
        self.spectrogram = np.repeat(self.spectrogram[..., np.newaxis], 3, axis=-1)
        print(self.spectrogram.shape)
        self.activation_matrix = midi_to_activation_matrix(midi_path)
        print(self.activation_matrix.shape)

        #Dopasowanie długości do krótszego
        min_frames = min(
            self.spectrogram.shape[1],
            self.activation_matrix.shape[1],
        )
        self.spectrogram = self.spectrogram[:, :min_frames, :]
        self.activation_matrix = self.activation_matrix[:, :min_frames]
        print(self.spectrogram.shape)
        print(self.activation_matrix.shape)

        self.current_segment_idx = 0  #Reset indeksu segmentu

# Model oparty na EfficientNet
class AMTModel(nn.Module):
    def __init__(self, num_notes=88, timesteps=224):
        super(AMTModel, self).__init__()
        # EfficientNet jako ekstraktor cech
        self.feature_extractor = efficientnet_b0(pretrained=True)
        self.feature_extractor.features = nn.Sequential(*list(self.feature_extractor.features)[:-1])  # Bez top
        self.feature_extractor.requires_grad_(False)

        # Dopasowanie wymiarów czasowych
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(1280 * 7 * 7, 224 * timesteps)

        # Warstwy LSTM
        self.lstm1 = nn.LSTM(input_size=224, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        # Wyjście
        self.output = nn.Linear(128, num_notes)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Złączenie batcha i ramek

        # Ekstrakcja cech
        x = self.feature_extractor(x)  # (batch_size * num_frames, 1280, 7, 7)
        x = self.flatten(x)  # (batch_size * num_frames, 1280 * 7 * 7)

        # Dopasowanie do czasowego wymiaru
        x = self.fc(x)  # (batch_size * num_frames, 224 * timesteps)
        x = x.view(batch_size, num_frames, -1)  # Przywrócenie batcha i ramek

        # Przejście przez LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Wyjście
        x = self.output(x)  # (batch_size, num_frames, num_notes)
        return x


# Podział danych
audio_paths = []
midi_paths = []

for file in os.listdir('train_model/wav/Mozart'):
    audio_paths.append(os.path.join('train_model/wav/Mozart', file))
    midi_paths.append(os.path.join('train_model/midi/Mozart', file[:-3] + 'midi'))

audio_train, audio_temp, midi_train, midi_temp = train_test_split(
    audio_paths, midi_paths, test_size=0.3, random_state=42
)
audio_val, audio_test, midi_val, midi_test = train_test_split(
    audio_temp, midi_temp, test_size=0.5, random_state=42
)

train_dataset = AMTDataset(audio_train, midi_train)
val_dataset = AMTDataset(audio_val, midi_val)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Model, loss i optymalizator
model = AMTModel().to(device)
summary(model)
#criterion = nn.BCELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
# def train_model(model, train_loader, val_loader, num_epochs=20):
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         val_loss = 0.0
#         model.eval()
#         with torch.no_grad():
#             for X, y in val_loader:
#                 X, y = X.to(device), y.to(device)
#                 outputs = model(X)
#                 loss = criterion(outputs, y)
#                 val_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


#train_model(model, train_loader, val_loader)
