import librosa
import torch
import random
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .telephony_effects import apply_telephony_effects
class AudioDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["path"]

        label = row["label"]

        # Load audio
        y, sr = librosa.load(path, sr=8000)
        # Apply telephony simulation
        y = apply_telephony_effects(y)
        # Avoid completely silent audio
        if np.max(np.abs(y)) > 0:
         y = y / np.max(np.abs(y))

        # Fixed length = 4 sec
        max_len = 16000 * 4

        # Pad or trim
        if len(y) < max_len:

            pad = max_len - len(y)

            y = np.pad(y, (0, pad))

        else:

            y = y[:max_len]

        # Create Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        # Convert to log scale
        mel = librosa.power_to_db(mel)
        mel_std = mel.std()
        if mel_std != 0:
           mel = (mel - mel.mean()) / mel_std
        else:
           mel = mel - mel.mean()

        # Convert to tensor
        mel = torch.tensor(mel).unsqueeze(0).float()

        label = torch.tensor(label).long()

        return mel, label