# Speech Deepfake Detection - Multi-Model Research Framework

A comprehensive PyTorch-based research framework for detecting deepfake audio using multiple state-of-the-art architectures. This project compares three models: CNN + Mel Spectrogram, ResNet18 + Mel Spectrogram, and Wav2Vec2.

## Project Overview

This framework implements and evaluates three distinct approaches to speech deepfake detection:

1. **CNN + Mel Spectrogram** - Baseline convolutional neural network on Mel frequency spectrograms
2. **ResNet18 + Mel Spectrogram** - Pretrained ResNet18 adapted for audio classification
3. **Wav2Vec2** - Self-supervised pretrained Wav2Vec2 model with fine-tuning

All models are trained with realistic telephony simulation effects to improve robustness.

## Key Features

### Enhanced Telephony Simulation
The framework includes sophisticated telephony simulation effects to create realistic training scenarios:

- **μ-law Codec Compression**: Simulates ITU-T G.711 codec artifacts with quantization
- **Packet Loss Simulation**: Randomly zeros out audio chunks to simulate VoIP packet drops
- **Reverberation/Room Impulse**: Adds realistic echo and room effects using convolution
- **Background Noise**: Adds Gaussian noise at varying levels
- **Volume Scaling**: Random dynamic range compression

These effects are randomly applied during training to create diverse, realistic audio degradations.

### Model Architectures

#### 1. CNN Baseline
- Simple 2-layer CNN with adaptive pooling
- Input: 128-dimensional Mel spectrograms (1 channel)
- Output: 2-class classification (real/fake)
- Features: Minimal baseline for comparison

#### 2. ResNet18
- Pretrained ImageNet weights with transfer learning
- Converts single-channel Mel spectrograms to 3-channel via repetition
- Custom classification head with dropout and batch normalization
- **Discriminative Learning Rates**: Lower LR for pretrained backbone (1e-5), higher for head (1e-4)

#### 3. Wav2Vec2
- Pretrained facebook/wav2vec2-base model
- **Partial Unfreezing**: Last 4 transformer layers unfrozen for fine-tuning
- Processes raw 16kHz audio waveforms (no spectrogram conversion needed)
- Custom classification head with dense layers
- **Discriminative Learning Rates**: Encoder LR 1e-5, classifier LR 1e-4

## Project Structure

```
.
├── src/
│   ├── model.py                 # CNN baseline model
│   ├── resnet_model.py          # ResNet18 implementation
│   ├── wav2vec_model.py         # Wav2Vec2 with partial unfreezing
│   ├── dataset_loader.py        # Mel spectrogram dataset loader
│   ├── wav2vec_dataset.py       # Raw audio dataset loader
│   ├── telephony_effects.py     # Enhanced telephony simulation
│   ├── extract_audio.py         # Audio extraction utilities
│   └── create_csv.py            # CSV generation utilities
├── train.py                     # CNN training script
├── train_resnet.py              # ResNet18 training script
├── train_wav2vec.py             # Wav2Vec2 training script
├── evaluate_models.py           # Evaluation and comparison pipeline
├── check_balance.py             # Dataset balance checking
├── test_dataset.py              # Dataset verification
├── README.md                    # This file
└── requirements_minimal.txt     # Dependencies

outputs/
├── model.pth                    # CNN checkpoint
├── cnn/                         # CNN artifacts
├── resnet/
│   └── best_model.pth           # ResNet18 checkpoint
├── wav2vec2/
│   └── best_model.pth           # Wav2Vec2 checkpoint
├── plots/
│   ├── roc_curves.png           # ROC comparison
│   └── confusion_matrices.png   # Confusion matrices
└── metrics/
    └── comparison_metrics.csv   # Final metrics table
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Install dependencies
pip install -r requirements_minimal.txt

# Or install individually
pip install torch torchvision librosa numpy pandas scipy scikit-learn matplotlib seaborn transformers
```

## Dataset Preparation

The framework expects a dataset with the following structure:

```
dataset/
└── metadata.csv  # CSV file with columns: [path, label]
                  # path: path to audio file
                  # label: 0 (real) or 1 (fake)
```

Example metadata.csv:
```
path,label
/path/to/real_audio_1.wav,0
/path/to/fake_audio_1.wav,1
/path/to/real_audio_2.wav,0
```

## Training

### Train CNN Baseline
```bash
python train.py
```
- Trains on Mel spectrograms with enhanced telephony effects
- Saves best model to `outputs/model.pth`
- Epoch-based training with validation monitoring

### Train ResNet18
```bash
python train_resnet.py
```
- Uses pretrained ResNet18 from torchvision
- Implements discriminative learning rates
- Learning rate scheduling based on validation F1
- Saves best model to `outputs/resnet/best_model.pth`
- Training: ~20 epochs (adaptive)

### Train Wav2Vec2
```bash
python train_wav2vec.py
```
- Uses pretrained facebook/wav2vec2-base
- Processes raw audio (no spectrogram needed)
- Partial unfreezing strategy for efficient fine-tuning
- Gradient clipping and early stopping
- Saves best model to `outputs/wav2vec2/best_model.pth`
- Training: ~15 epochs (with early stopping)

## Evaluation

### Run Comprehensive Evaluation
```bash
python evaluate_models.py
```

This script:
1. Loads all three trained models
2. Evaluates on test set
3. Computes all metrics
4. Generates visualizations
5. Produces comparison table

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Anti-Spoofing Metrics (Most Important)
- **EER (Equal Error Rate)**: The point where FAR = FRR
  - Lower EER indicates better discrimination between real and fake
  - Industry standard for anti-spoofing systems
- **FAR (False Accept Rate)**: Fake samples accepted as real
- **FRR (False Reject Rate)**: Real samples rejected as fake

## Telephony Simulation Details

### μ-law Companding
Simulates G.711 codec compression used in telephony:
```
- Compression ratio: reduces quality with 8-bit quantization
- Applied probability: 70% during training
- Realistic bitrate artifact simulation
```

### Packet Loss
Simulates VoIP network degradation:
```
- Loss probability: 2-10% random
- Packet duration: 20-40ms configurable
- Zero-ing strategy: Simulates dropped RTP packets
- Applied probability: 50% during training
```

### Reverberation
Adds room acoustics via impulse response convolution:
```
- Room scale: 0.1-0.4 (mild reverberation)
- Delay range: 800-1600 samples
- Multi-reflection modeling
- Applied probability: 60% during training
```

## Configuration

### Hyperparameters

| Parameter | CNN | ResNet18 | Wav2Vec2 |
|-----------|-----|----------|---------|
| Batch Size | 16 | 16 | 8 |
| Learning Rate (Encoder) | 1e-5 | 1e-5 | 1e-5 |
| Learning Rate (Head) | - | 1e-4 | 1e-4 |
| Dropout | - | 0.5 | 0.3 |
| Epochs | 10 | 20 | 15 |
| Optimizer | Adam | AdamW | AdamW |
| Weight Decay | - | 1e-4 | 1e-4 |
| Gradient Clipping | - | 1.0 | 1.0 |

### Data Augmentation (Telephony)
All models use the same telephony augmentation pipeline:
- Volume scaling: 60-100%
- Codec compression: 70% probability
- Packet loss: 50% probability
- Reverberation: 60% probability
- Gaussian noise: 80% probability

## Outputs

After evaluation, the following files are generated:

### Metrics (outputs/metrics/)
- `comparison_metrics.csv`: Tab-separated comparison of all metrics

### Visualizations (outputs/plots/)
- `roc_curves.png`: ROC curve comparison (all three models)
- `confusion_matrices.png`: Confusion matrices side-by-side

### Model Checkpoints
- `outputs/model.pth`: CNN best model
- `outputs/resnet/best_model.pth`: ResNet18 best model
- `outputs/wav2vec2/best_model.pth`: Wav2Vec2 best model

## Performance Recommendations

### Hardware Requirements
- GPU: NVIDIA GPU with CUDA 11.8+ recommended
  - ResNet18: ~4GB VRAM
  - Wav2Vec2: ~6-8GB VRAM
- CPU: Can run on CPU but significantly slower
- RAM: 8GB+ recommended

### Optimization Tips
1. **Batch Size Tuning**: Adjust based on available GPU memory
2. **Learning Rates**: Discriminative LR significantly improves transfer learning
3. **Early Stopping**: Enable to prevent overfitting
4. **Data Augmentation**: Telephony effects are critical for robustness
5. **Gradient Accumulation**: For very large batches on small GPUs

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch size in training scripts
- Enable gradient checkpointing (requires code modification)
- Reduce model size or audio length

### NaN Loss During Training
- Check audio normalization (should be in [-1, 1])
- Verify label format (0 or 1)
- Reduce learning rate
- Check for silent audio files

### Model Not Improving
- Verify telephony effects are being applied
- Check class balance with `python check_balance.py`
- Increase training epochs
- Verify training data quality

## References

- Wav2Vec2: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- μ-law Companding: [ITU-T G.711](https://www.itu.int/rec/T-REC-G.711/199209-W/en)

## License

[Specify your license here]

## Authors

- Research team at [Your Institution]
- Enhanced with multi-model framework and telephony simulation

---

**Last Updated**: June 2026
**Framework Version**: 2.0 (Multi-Model Comparative)
