import librosa
import torch
import random
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def apply_telephony_effects(y):
    """
    Apply telephony simulation effects (SAME as CNN pipeline).
    """
    # Random volume scaling
    volume_scale = random.uniform(0.7, 1.0)
    y = y * volume_scale

    # Add Gaussian background noise
    noise = np.random.normal(0, 0.005, len(y))
    y = y + noise

    # Clip values to [-1, 1]
    y = np.clip(y, -1.0, 1.0)

    return y


class Wav2Vec2Dataset(Dataset):
    """
    Dataset loader for Wav2Vec2 model.
    
    - Loads audio at 8kHz (same as CNN)
    - Applies telephony effects (same as CNN)
    - Pads/truncates to fixed 4-second duration
    - Outputs raw waveform tensor
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["label"]

        # Load audio at 8kHz (same as CNN)
        y, sr = librosa.load(path, sr=8000)

        # Apply telephony effects (same as CNN)
        y = apply_telephony_effects(y)

        # Normalize to prevent clipping
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Fixed length = 4 sec at 8kHz = 32000 samples
        max_len = 8000 * 4

        # Pad or truncate
        if len(y) < max_len:
            pad = max_len - len(y)
            y = np.pad(y, (0, pad))
        else:
            y = y[:max_len]

        # Convert to tensor (raw waveform, not spectrogram)
        waveform = torch.tensor(y).float()

        label = torch.tensor(label).long()

        return waveform, label
