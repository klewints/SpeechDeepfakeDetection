# Speech Deepfake Detection - Dual Pipeline Implementation

## Overview

This project implements **two parallel pipelines** for binary speech deepfake detection:

1. **CNN + Mel Spectrogram Pipeline** (Original)
2. **Wav2Vec2 Pipeline** (New)

Both pipelines use the same **telephony effects preprocessing** to simulate real-world noisy speech conditions.

---

## Project Structure

```
.
├── src/
│   ├── dataset_loader.py        # CNN dataset loader (Mel Spectrogram)
│   ├── wav2vec_dataset.py       # Wav2Vec2 dataset loader (raw waveform)
│   ├── model.py                 # CNN model (unchanged)
│   ├── wav2vec_model.py         # Wav2Vec2 classifier (new)
│   ├── extract_audio.py         # Audio extraction
│   └── create_csv.py            # CSV metadata creation
├── train.py                      # Train CNN model
├── train_wav2vec.py             # Train Wav2Vec2 model (new)
├── evaluate_models.py           # Evaluate & compare both models (new)
├── requirements_updated.txt     # Updated dependencies with transformers
├── dataset/
│   ├── metadata.csv             # Audio paths and labels
│   └── audio/
│       ├── real/                # Real audio files
│       └── fake/                # Deepfake audio files
└── outputs/
    ├── model.pth                       # Trained CNN model
    ├── wav2vec2_model_best.pth        # Best Wav2Vec2 model
    ├── confusion_matrix_CNN_Mel.png
    ├── confusion_matrix_Wav2Vec2.png
    ├── roc_curves_comparison.png
    └── evaluation_results.csv
```

---

## Preprocessing Pipeline

### Shared Telephony Effects (Both Models)

All audio goes through the same preprocessing:

```python
def apply_telephony_effects(y):
    # 1. Random volume scaling (0.7 - 1.0x)
    volume_scale = random.uniform(0.7, 1.0)
    y = y * volume_scale
    
    # 2. Add Gaussian background noise (μ=0, σ=0.005)
    noise = np.random.normal(0, 0.005, len(y))
    y = y + noise
    
    # 3. Clip to [-1.0, 1.0] to prevent clipping artifacts
    y = np.clip(y, -1.0, 1.0)
    
    return y
```

### Model-Specific Processing

#### CNN Pipeline
```
Audio (any duration)
  ↓
Load at 8kHz (8000 samples/second)
  ↓
Apply telephony effects
  ↓
Normalize
  ↓
Pad/truncate to 4 seconds (32,000 samples)
  ↓
Compute Mel Spectrogram (128 bins)
  ↓
Normalize to log-scale (dB)
  ↓
Return as Tensor(1, 128, T)
```

#### Wav2Vec2 Pipeline
```
Audio (any duration)
  ↓
Load at 8kHz (8000 samples/second)
  ↓
Apply telephony effects (SAME)
  ↓
Normalize
  ↓
Pad/truncate to 4 seconds (32,000 samples)
  ↓
Return raw waveform as Tensor(32,000)
  ↓
Wav2Vec2 encodes to contextualized features
```

---

## Model Architectures

### CNN + Mel Spectrogram

```
Input: (1, 128, T) Mel Spectrogram
  ↓
Conv2d(1→16, 3×3) + ReLU + MaxPool2d
  ↓
Conv2d(16→32, 3×3) + ReLU + MaxPool2d
  ↓
AdaptiveAvgPool2d → (8×8)
  ↓
Flatten → 2048
  ↓
Linear(2048→128) + ReLU
  ↓
Linear(128→2)
  ↓
Output: Logits [real, fake]
```

### Wav2Vec2 Classifier

```
Input: (32,000) Raw Waveform at 8kHz
  ↓
Wav2Vec2Model (frozen encoder)
  - Extracts contextualized speech features
  - Output shape: (T, 768)
  ↓
Mean Pooling over time: → (768,)
  ↓
Linear(768→128) + ReLU + Dropout(0.3)
  ↓
Linear(128→2)
  ↓
Output: Logits [real, fake]
```

---

## Training

### CNN Training (Original)

```bash
python train.py
```

**Configuration:**
- Epochs: 10
- Batch size: 16
- Optimizer: Adam (lr=1e-5)
- Loss: CrossEntropyLoss
- Class balancing: WeightedRandomSampler
- Train/Val split: 80/20

**Output:** `outputs/model.pth`

### Wav2Vec2 Training (New)

```bash
python train_wav2vec.py
```

**Configuration:**
- Epochs: 15
- Batch size: 8 (smaller due to larger model)
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Class balancing: WeightedRandomSampler
- Train/Val split: 80/20
- Early stopping: 3 epochs patience on validation F1
- Encoder: Frozen (efficient transfer learning)

**Output:** 
- `outputs/wav2vec2_model_best.pth` (best validation F1)
- `outputs/wav2vec2_model_final.pth` (final epoch)

---

## Evaluation

### Comprehensive Metrics

Both models are evaluated on the FULL dataset using:

1. **Accuracy**: (TP + TN) / (Total)
2. **Precision**: TP / (TP + FP) — False positive rate
3. **Recall**: TP / (TP + FN) — False negative rate
4. **F1 Score**: Harmonic mean of Precision and Recall
5. **ROC-AUC**: Area under the ROC curve
6. **EER**: Equal Error Rate (FAR = FRR) — **MOST IMPORTANT**
7. **Confusion Matrix**: TN, FP, FN, TP counts

### Run Evaluation

```bash
python evaluate_models.py
```

**Outputs:**
- `outputs/confusion_matrix_CNN_Mel.png` — Confusion matrix visualization
- `outputs/confusion_matrix_Wav2Vec2.png` — Confusion matrix visualization
- `outputs/roc_curves_comparison.png` — ROC curves for both models
- `outputs/evaluation_results.csv` — Detailed metrics table

---

## Equal Error Rate (EER)

EER is computed by finding the threshold where **False Acceptance Rate (FAR) = False Rejection Rate (FRR)**.

```
FAR = FP / (FP + TN)  — False positives (fake predicted as real)
FRR = FN / (FN + TP)  — False negatives (real predicted as fake)
```

The EER threshold is the point where these rates intersect. **Lower EER is better.**

**Example:**
- CNN EER: 0.0845 (8.45% error rate)
- Wav2Vec2 EER: 0.0612 (6.12% error rate)

---

## Dataset Labels

Binary classification:
- **Label 0**: Real speech
- **Label 1**: Deepfake speech

**Class distribution:**
- Train set: 80% of data (with weighted sampling)
- Validation set: 20% of data
- Test: Entire dataset for final evaluation

---

## Expected Results

Both models should achieve:
- **Accuracy:** >90%
- **Precision:** >85%
- **Recall:** >85%
- **F1 Score:** >85%
- **ROC-AUC:** >0.95
- **EER:** <10% (ideally <5%)

Wav2Vec2 typically outperforms CNN because:
- Uses pretrained contextualized speech representations
- Captures phonetic and speaker information
- Robust to acoustic variations

---

## Installation & Dependencies

```bash
# Install dependencies
pip install -r requirements_updated.txt

# Key packages:
# - torch (PyTorch)
# - transformers (HuggingFace, includes Wav2Vec2)
# - scikit-learn (Metrics, ROC curves)
# - librosa (Audio processing)
# - matplotlib (Visualization)
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)
- Reduce batch size in `train_wav2vec.py` (line 81: `batch_size=4` or `batch_size=2`)
- The Wav2Vec2 model is ~360MB; freeze encoder to reduce memory

### Transformers Model Download
- First time running will download Wav2Vec2 (~370MB)
- Cached in `~/.cache/huggingface/hub/`
- Internet connection required

### Dataset Not Found
- Ensure `dataset/metadata.csv` exists
- Ensure audio files are in `dataset/audio/real/` and `dataset/audio/fake/`
- Run `src/extract_audio.py` and `src/create_csv.py` first

### NaN Loss
- Usually indicates audio normalization issues
- Check that audio files are valid WAV files
- Verify sample rate is 8kHz after loading

---

## Key Improvements in Wav2Vec2

| Aspect | CNN | Wav2Vec2 |
|--------|-----|----------|
| Input | Mel Spectrogram (128×T) | Raw waveform (32K,) |
| Feature Learning | Fixed (librosa) | Learned (pretrained) |
| Contextual Info | Limited | Excellent (transformer) |
| Efficiency | Fast inference (~10ms) | Slower (~100ms) |
| Robustness | Moderate | High |
| Deepfake Detection | Good | Excellent |

---

## Citation & References

- **Wav2Vec2**: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)
  - Unsupervised speech representation learning using contrastive learning
  - Pretrained on 53k hours of unlabeled speech
  
- **HuggingFace Transformers**: [Wolf et al., 2020](https://aclanthology.org/2020.emnlp-main.552/)

---

## Next Steps

To extend this project:

1. **Fine-tune Wav2Vec2 encoder** — Unfreeze and train end-to-end
2. **Ensemble methods** — Combine CNN and Wav2Vec2 predictions
3. **Data augmentation** — Add pitch shift, time stretch, speaker variation
4. **Cross-validation** — Use k-fold for more robust evaluation
5. **Explainability** — Add attention visualization (LIME, GradCAM)

---

## Authors

- Original CNN pipeline: [Previous implementation]
- Wav2Vec2 extension: [Current implementation]
- Evaluation framework: [Comprehensive metrics & visualization]

---

**Last Updated:** 2026-06-19
