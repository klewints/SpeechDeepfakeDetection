"""
Wav2Vec2 dataset loader for raw audio input.

Wav2Vec2 works directly on raw audio waveforms, not spectrograms.
This loader provides raw audio with telephony effects applied.
"""

import librosa
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .telephony_effects import apply_telephony_effects


class AudioRawDataset(Dataset):
    """
    Raw audio dataset for Wav2Vec2 model.
    
    Loads audio at 16kHz (standard for most pretrained Wav2Vec2 models)
    and applies telephony effects for data augmentation.
    """
    
    def __init__(self, csv_file, target_sr=16000):
        """
        Initialize dataset.
        
        Args:
            csv_file: Path to CSV with columns ['path', 'label']
            target_sr: Target sampling rate (default 16kHz for Wav2Vec2)
        """
        self.df = pd.read_csv(csv_file)
        self.target_sr = target_sr
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get item.
        
        Returns:
            (waveform, label) tuple
            - waveform: (1, max_len) - normalized raw audio
            - label: 0 (real) or 1 (fake)
        """
        row = self.df.iloc[idx]
        path = row["path"]
        label = row["label"]
        
        # Load audio at target sampling rate
        y, sr = librosa.load(path, sr=self.target_sr)
        
        # Apply telephony effects
        y = apply_telephony_effects(y, sr=self.target_sr)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Fixed length = 4 seconds at 16kHz = 64000 samples
        max_len = self.target_sr * 4
        
        # Pad or trim
        if len(y) < max_len:
            pad = max_len - len(y)
            y = np.pad(y, (0, pad))
        else:
            y = y[:max_len]
        
        # Convert to tensor (1, max_len) format
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return y, label
