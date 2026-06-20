# Speech Deepfake Detection for Telephonic Audio

## Overview

This project focuses on detecting deepfake speech in telephonic conditions using three different deep learning approaches:

1. CNN + Mel Spectrogram
2. ResNet18 + Mel Spectrogram
3. Wav2Vec2

The goal was to compare traditional spectrogram-based models with transformer-based speech models on noisy telephone-style audio.

The project uses telephony simulation effects such as:

* background noise
* packet loss
* μ-law codec compression
* reverberation
* volume scaling

to better simulate real-world call conditions.

---

# Dataset

Dataset used:

* HAV-DF (Hindi Audio-Visual Deepfake Dataset)

The videos were converted into audio files and divided into:

* real audio
* fake audio

Metadata was stored in a CSV file containing:

* file path
* label

Labels:

* `0` → Real
* `1` → Fake

---

# Project Structure

```bash
.
├── dataset/
│   ├── audio/
│   │   ├── real/
│   │   └── fake/
│   └── metadata.csv
│
├── src/
│   ├── dataset_loader.py
│   ├── wav2vec_dataset.py
│   ├── model.py
│   ├── resnet_model.py
│   ├── wav2vec_model.py
│   ├── telephony_effects.py
│   ├── extract_audio.py
│   └── create_csv.py
│
├── outputs/
│   ├── model.pth
│   ├── plots/
│   └── metrics/
│
├── train.py
├── train_resnet.py
├── train_wav2vec.py
├── evaluate_models.py
└── requirements.txt
```

---

# Preprocessing

All models use the same telephony simulation pipeline before training.

Effects applied:

* μ-law codec compression
* packet loss simulation
* reverberation
* background noise
* random volume scaling

This was done to simulate speech received over mobile calls or VoIP systems.

---

# Models Used

## 1. CNN + Mel Spectrogram

Pipeline:

```text
Audio
→ Mel Spectrogram
→ CNN
→ Classification
```

The CNN model uses convolutional layers on mel spectrograms extracted from audio.

---

## 2. ResNet18 + Mel Spectrogram

Pipeline:

```text
Audio
→ Mel Spectrogram
→ ResNet18
→ Classification
```

ResNet18 is a deeper convolutional architecture with residual connections that help preserve information across layers.

---

## 3. Wav2Vec2

Pipeline:

```text
Raw Audio Waveform
→ Wav2Vec2 Transformer Encoder
→ Classification Head
```

Unlike CNN-based approaches, Wav2Vec2 learns directly from raw audio waveforms instead of manually created spectrograms.

---

# Training

## CNN / ResNet

* Batch size: 16
* Optimizer: Adam
* Loss: CrossEntropyLoss
* Input: Mel Spectrograms

## Wav2Vec2

* Batch size: 8
* Optimizer: Adam
* Input: Raw waveforms
* Pretrained transformer encoder used for transfer learning

---

# Evaluation Metrics

The following metrics were used for evaluation:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* EER (Equal Error Rate)

EER was considered the most important metric because it is commonly used in spoof/deepfake detection systems.

---

# Final Results

| Metric    | CNN + Mel | ResNet18 + Mel | Wav2Vec2 |
| --------- | --------- | -------------- | -------- |
| Accuracy  | 68.20%    | 68.48%         | 58.62%   |
| Precision | 73.17%    | 71.91%         | 60.39%   |
| Recall    | 68.09%    | 68.64%         | 58.22%   |
| F1 Score  | 69.16%    | 69.55%         | 58.09%   |
| ROC-AUC   | 0.6993    | 0.7411         | 0.6464   |
| EER       | 0.2891    | 0.3318         | 0.4204   |

---

# Observations

* ResNet18 performed best overall among the three models.
* CNN achieved high recall but produced more false positives.
* Wav2Vec2 performed reasonably despite the small dataset size.
* Telephony simulation helped make training closer to real-world conditions.

---

# Running the Project

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train CNN

```bash
python train.py
```

---

## Train ResNet18

```bash
python train_resnet.py
```

---

## Train Wav2Vec2

```bash
python train_wav2vec.py
```

---

## Evaluate Models

```bash
python evaluate_models.py
```

This generates:

* ROC curves
* confusion matrices
* metric comparison CSV

---

# Libraries Used

* PyTorch
* Transformers (HuggingFace)
* Librosa
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

# References

* Wav2Vec2 Paper: https://arxiv.org/abs/2006.11477
* HuggingFace Transformers: https://huggingface.co/docs/transformers/index
