"""
Wav2Vec2 training script with enhanced features.

Features:
- Partial unfreezing of transformer layers
- Discriminative learning rates
- Longer training with early stopping
- Gradient clipping and regularization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score
import os
from transformers import Wav2Vec2Processor
from src.wav2vec_dataset import AudioRawDataset
from src.wav2vec_model import Wav2Vec2DeepfakeModel


def pad_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Pads sequences to the maximum length in the batch.
    """
    waveforms, labels = zip(*batch)
    
    # Get max length
    max_len = max(w.shape[-1] for w in waveforms)
    
    # Pad waveforms
    padded_waveforms = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded_waveforms.append(w.squeeze(0))
    
    waveforms = torch.stack(padded_waveforms)
    labels = torch.stack(labels)
    
    return waveforms, labels


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    all_preds = []
    all_labels = []
    
    for input_values, labels in train_loader:
        input_values = input_values.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_values)
        loss = criterion(logits, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        
        _, predicted = torch.max(logits, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    train_loss_avg = train_loss / len(train_loader)
    
    return train_loss_avg, train_acc, train_f1


def validate_epoch(model, val_loader, criterion, device):
    """
    Validation for one epoch.
    """
    model.eval()
    
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_values, labels in val_loader:
            input_values = input_values.to(device)
            labels = labels.to(device)
            
            logits = model(input_values)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_loss_avg = val_loss / len(val_loader)
    
    return val_loss_avg, val_acc, val_f1


# =========================
# SETUP
# =========================

print("=" * 60)
print("Wav2Vec2 Training for Speech Deepfake Detection")
print("=" * 60)

# Create output directory
os.makedirs("outputs/wav2vec2", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

print("\nLoading dataset (raw audio)...")
dataset = AudioRawDataset("dataset/metadata.csv", target_sr=16000)
labels = dataset.df["label"].values

print(f"Total samples: {len(dataset)}")

# =========================
# TRAIN / VALIDATION SPLIT
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

# =========================
# CLASS BALANCING
# =========================

train_indices = train_dataset.indices
train_labels = [
    dataset.df.iloc[idx]["label"]
    for idx in train_indices
]

class_counts = np.bincount(train_labels)
print(f"Class distribution: {class_counts}")

class_weights = 1. / class_counts
sample_weights = [
    class_weights[label]
    for label in train_labels
]

sampler = WeightedRandomSampler(
    sample_weights,
    len(sample_weights)
)

# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,  # Smaller batch for GPU memory
    sampler=sampler,
    collate_fn=pad_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=pad_collate_fn
)

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# MODEL
# =========================

print("Loading Wav2Vec2 model...")
model = Wav2Vec2DeepfakeModel(
    unfreeze_layers=4,  # Unfreeze last 4 transformer layers
    dropout_rate=0.3,
    num_classes=2
).to(device)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# =========================
# LOSS FUNCTION
# =========================

criterion = nn.CrossEntropyLoss()

# =========================
# OPTIMIZER - DISCRIMINATIVE LEARNING RATES
# =========================

# Collect parameters by layer
encoder_params = []
head_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'classifier' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4}
], weight_decay=1e-4)

# =========================
# LEARNING RATE SCHEDULER
# =========================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

# =========================
# TRAINING LOOP
# =========================

epochs = 10
best_val_f1 = 0
best_model_path = "outputs/wav2vec2/best_model.pth"
early_stopping_patience = 5
patience_counter = 0

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

for epoch in range(epochs):
    
    train_loss, train_acc, train_f1 = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    val_loss, val_acc, val_f1 = validate_epoch(
        model, val_loader, criterion, device
    )
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_f1)
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break

# =========================
# FINAL RESULTS
# =========================

print("\n" + "=" * 60)
print(f"Training complete! Best F1: {best_val_f1:.4f}")
print(f"Model saved to: {best_model_path}")
print("=" * 60)
