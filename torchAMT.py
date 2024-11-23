import os
import numpy as np
import librosa
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
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


# Dataset dla PyTorcha
class AMTDataset(Dataset):
    def __init__(self, audio_paths, midi_paths, sr=22050, hop_length=512):
        self.audio_paths = audio_paths
        self.midi_paths = midi_paths
        self.sr = sr
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        midi_path = self.midi_paths[idx]

        spectrogram = generate_spectrogram(audio_path, sr=self.sr, hop_length=self.hop_length)
        spectrogram = np.repeat(spectrogram[..., np.newaxis], 3, axis=-1)  # Dopasowanie do RGB

        activation_matrix = midi_to_activation_matrix(midi_path, sr=self.sr, hop_length=self.hop_length)
        num_frames = min(spectrogram.shape[1] // 224, activation_matrix.shape[1] // 224)

        x = []
        y = []
        for j in range(num_frames):
            x.append(spectrogram[:, j * 224:(j + 1) * 224, :])
            y.append(activation_matrix[:, j * 224:(j + 1) * 224])

        return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model, loss i optymalizator
model = AMTModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie modelu
def train_model(model, train_loader, val_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


train_model(model, train_loader, val_loader)
