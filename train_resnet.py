"""
ResNet18 training script for speech deepfake detection.

Uses Mel spectrograms with enhanced telephony simulation.
Implements learning rate scheduling, best model saving, and F1-based validation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
from src.dataset_loader import AudioDataset
from src.resnet_model import ResNet18MelModel


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
    
    Returns:
        Average training loss, accuracy, F1 score
    """
    model.train()
    
    train_loss = 0
    train_correct = 0
    train_total = 0
    all_preds = []
    all_labels = []
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    train_loss_avg = train_loss / len(train_loader)
    
    return train_loss_avg, train_acc, train_f1


def validate_epoch(model, val_loader, criterion, device):
    """
    Validation for one epoch.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: torch.device
    
    Returns:
        Average validation loss, accuracy, F1 score
    """
    model.eval()
    
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_loss_avg = val_loss / len(val_loader)
    
    return val_loss_avg, val_acc, val_f1


# =========================
# SETUP
# =========================

print("=" * 60)
print("ResNet18 Training for Speech Deepfake Detection")
print("=" * 60)

# Create output directory
os.makedirs("outputs/resnet", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

print("\nLoading dataset...")
dataset = AudioDataset("dataset/metadata.csv")
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
# CLASS BALANCING WITH WEIGHTED SAMPLING
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
    batch_size=16,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# =========================
# MODEL
# =========================

print("Loading pretrained ResNet18...")
model = ResNet18MelModel(pretrained=True, dropout_rate=0.5).to(device)

# =========================
# LOSS FUNCTION
# =========================

criterion = nn.CrossEntropyLoss()

# =========================
# OPTIMIZER - LOWER LR FOR PRETRAINED BACKBONE
# =========================

# Use discriminative learning rates
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if 'resnet.fc' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-4}
], weight_decay=1e-4)

# =========================
# LEARNING RATE SCHEDULER
# =========================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# =========================
# TRAINING LOOP
# =========================

epochs = 20
best_val_f1 = 0
best_model_path = "outputs/resnet/best_model.pth"

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
    
    # Save best model based on F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (F1: {val_f1:.4f})")

# =========================
# FINAL MODEL SAVE
# =========================

print("\n" + "=" * 60)
print(f"Training complete! Best F1: {best_val_f1:.4f}")
print(f"Model saved to: {best_model_path}")
print("=" * 60)
