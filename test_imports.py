"""
Integration test to verify all modules can be imported successfully.
This does not require training data to be present.
"""

import sys

print("Testing module imports...")
print("=" * 60)

# Test core dependencies
try:
    import torch
    import numpy as np
    import pandas as pd
    print("✓ Core dependencies (torch, numpy, pandas)")
except ImportError as e:
    print(f"✗ Core dependencies failed: {e}")
    sys.exit(1)

# Test audio processing
try:
    import librosa
    from scipy import signal
    print("✓ Audio processing (librosa, scipy)")
except ImportError as e:
    print(f"✗ Audio processing failed: {e}")
    sys.exit(1)

# Test ML/visualization
try:
    from sklearn.metrics import f1_score, roc_auc_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ ML and visualization (sklearn, matplotlib, seaborn)")
except ImportError as e:
    print(f"✗ ML and visualization failed: {e}")
    sys.exit(1)

# Test transformers
try:
    from transformers import Wav2Vec2Model
    print("✓ Transformers library")
except ImportError as e:
    print(f"✗ Transformers failed: {e}")
    sys.exit(1)

# Test custom modules
print("\nTesting custom modules...")

try:
    from src.telephony_effects import apply_telephony_effects, apply_mulaw_companding
    print("✓ telephony_effects module")
except ImportError as e:
    print(f"✗ telephony_effects failed: {e}")
    sys.exit(1)

try:
    from src.model import CNNModel
    print("✓ CNN model")
except ImportError as e:
    print(f"✗ CNN model failed: {e}")
    sys.exit(1)

try:
    from src.resnet_model import ResNet18MelModel
    print("✓ ResNet18 model")
except ImportError as e:
    print(f"✗ ResNet18 model failed: {e}")
    sys.exit(1)

try:
    from src.wav2vec_model import Wav2Vec2DeepfakeModel
    print("✓ Wav2Vec2 model")
except ImportError as e:
    print(f"✗ Wav2Vec2 model failed: {e}")
    sys.exit(1)

try:
    from src.wav2vec_dataset import AudioRawDataset
    print("✓ Wav2Vec2 dataset loader")
except ImportError as e:
    print(f"✗ Wav2Vec2 dataset failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All imports successful! Framework is ready.")
print("=" * 60)
