import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc
from src.dataset_loader import AudioDataset
from src.model import CNNModel


def compute_eer(fpr, fnr):
    """
    Compute Equal Error Rate (EER) from FPR and FNR curves.
    EER is the point where FPR = FNR.
    
    Args:
        fpr: False positive rates
        fnr: False negative rates
    
    Returns:
        eer: Equal error rate value
        threshold_idx: Index where EER occurs
    """
    diff = np.abs(fpr - fnr)
    threshold_idx = np.argmin(diff)
    eer = (fpr[threshold_idx] + fnr[threshold_idx]) / 2
    return eer, threshold_idx

# =========================
# LOAD DATASET
# =========================

dataset = AudioDataset(
    "dataset/metadata.csv"
)
labels = dataset.df["label"].values

print("Total samples:", len(dataset))

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
# DATALOADERS
# =========================
# =========================
# GET TRAIN LABELS ONLY
# =========================

train_indices = train_dataset.indices

train_labels = [
    dataset.df.iloc[idx]["label"]
    for idx in train_indices
]

# =========================
# CLASS BALANCING
# =========================

class_counts = np.bincount(train_labels)

class_weights = 1. / class_counts

sample_weights = [
    class_weights[label]
    for label in train_labels
]

sampler = WeightedRandomSampler(
    sample_weights,
    len(sample_weights)
)
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

print("Using device:", device)

# =========================
# MODEL
# =========================

model = CNNModel().to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5
)

# =========================
# TRAINING LOOP
# =========================

epochs = 10
best_val_eer = float('inf')
best_val_roc_auc = 0
best_model_path = "outputs/best_model.pth"

for epoch in range(epochs):

    # ======================
    # TRAINING
    # ======================

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0
    train_preds = []
    train_labels = []

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)
        
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(train_labels, train_preds, average='weighted')

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0
    val_preds = []
    val_labels = []
    val_probs = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()

            # Get probabilities for positive class
            probs = torch.softmax(outputs, dim=1)
            val_probs.extend(probs[:, 1].cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y).sum().item()
            val_total += y.size(0)
            
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    val_roc_auc = auc(fpr, tpr)
    
    # Compute EER
    fnr = 1 - tpr
    val_eer, _ = compute_eer(fpr, fnr)

    # ======================
    # PRINT RESULTS
    # ======================

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
    print(f"  Val EER:    {val_eer:.4f} | ROC-AUC: {val_roc_auc:.4f}")
    
    # Save best model based on EER (lower is better)
    # If EER is equal, use higher ROC-AUC as tie-breaker
    if val_eer < best_val_eer or (val_eer == best_val_eer and val_roc_auc > best_val_roc_auc):
        best_val_eer = val_eer
        best_val_roc_auc = val_roc_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (EER: {val_eer:.4f}, ROC-AUC: {val_roc_auc:.4f})")

# =========================
# FINAL RESULTS
# =========================

print("\n" + "=" * 60)
print(f"Training complete! Best EER: {best_val_eer:.4f}, Best ROC-AUC: {best_val_roc_auc:.4f}")
print(f"Model saved to: {best_model_path}")
print("=" * 60)