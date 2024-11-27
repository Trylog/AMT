import os
import torch
import torchaudio
import numpy as np
import pretty_midi
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
from torchvision import models
from sklearn.model_selection import train_test_split

# Sprawdzenie dostępności GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uruchamianie na: {device}")


# Generowanie log-mel spektrogramu
def generate_spectrogram(audio_path, sr=44100, n_mels=224, hop_length=512):
    waveform, sr = torchaudio.load(audio_path)
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )(waveform)
    log_spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    return log_spectrogram.squeeze(0).numpy()


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


# Dataset dla PyTorch
class MusicTranscriptionDataset(Dataset):
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
        activation_matrix = midi_to_activation_matrix(midi_path, sr=self.sr, hop_length=self.hop_length)

        min_length = min(spectrogram.shape[1], activation_matrix.shape[1])
        spectrogram = spectrogram[:, :min_length]
        activation_matrix = activation_matrix[:, :min_length]

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # Kanał dodatkowy
        activation_matrix = torch.tensor(activation_matrix, dtype=torch.float32)
        return spectrogram, activation_matrix


# Model
class MusicTranscriptionModel(nn.Module):
    def __init__(self, num_notes=88, timesteps=224):
        super(MusicTranscriptionModel, self).__init__()

        self.feature_extractor = models.efficientnet_b0(pretrained=True)
        self.feature_extractor.features = nn.Sequential(*list(self.feature_extractor.features.children())[:-1])
        self.lstm1 = nn.LSTM(1280, 256, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_notes)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.feature_extractor(x)  # Wyjście: (batch, 1280, H, W)
        x = x.view(batch_size, -1, 1280)  # Dopasowanie do LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return torch.sigmoid(x.permute(0, 2, 1))  # Dopasowanie do (batch, notes, time)


# Przygotowanie danych

audio_paths = []
midi_paths = []

for file in os.listdir('train_model/wav/Mozart'):
    audio_paths.append(os.path.join('train_model/wav/Mozart', file))
    midi_paths.append(os.path.join('train_model/midi/Mozart', file[:-3] + 'midi'))
for file in os.listdir('train_model/wav/Beethoven'):
    audio_paths.append(os.path.join('train_model/wav/Beethoven', file))
    midi_paths.append(os.path.join('train_model/midi/Beethoven', file[:-3] + 'midi'))
for file in os.listdir('train_model/wav/Haydn'):
    audio_paths.append(os.path.join('train_model/wav/Haydn', file))
    midi_paths.append(os.path.join('train_model/midi/Haydn', file[:-3] + 'midi'))

# Podział na zestawy
audio_train, audio_temp, midi_train, midi_temp = train_test_split(
    audio_paths, midi_paths, test_size=0.3, random_state=42
)
audio_val, audio_test, midi_val, midi_test = train_test_split(
    audio_temp, midi_temp, test_size=0.5, random_state=42
)

train_dataset = MusicTranscriptionDataset(audio_train, midi_train)
val_dataset = MusicTranscriptionDataset(audio_val, midi_val)
test_dataset = MusicTranscriptionDataset(audio_test, midi_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Model i optymalizacja
model = MusicTranscriptionModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)


# Trenowanie

def train_model(model, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for spectrograms, activation_matrices in train_loader:
            spectrograms, activation_matrices = spectrograms.to(device), activation_matrices.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, activation_matrices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for spectrograms, activation_matrices in val_loader:
                spectrograms, activation_matrices = spectrograms.to(device), activation_matrices.to(device)
                outputs = model(spectrograms)
                loss = criterion(outputs, activation_matrices)
                val_loss += loss.item()

        scheduler.step(val_loss)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('spawn')

    # Wywołanie kodu głównego
    train_model(model, train_loader, val_loader)