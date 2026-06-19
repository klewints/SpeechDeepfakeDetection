import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score
import os

from src.wav2vec_dataset import Wav2Vec2Dataset
from src.wav2vec_model import Wav2Vec2Classifier


# =========================
# CREATE OUTPUT DIRECTORY
# =========================
os.makedirs("outputs", exist_ok=True)


# =========================
# LOAD DATASET
# =========================
print("Loading Wav2Vec2 dataset...")
dataset = Wav2Vec2Dataset("dataset/metadata.csv")
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
# GET TRAIN LABELS FOR CLASS BALANCING
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
class_weights = 1.0 / class_counts

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
    batch_size=8,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# MODEL
# =========================
print("Loading Wav2Vec2 model (facebook/wav2vec2-base)...")
model = Wav2Vec2Classifier(
    model_name="facebook/wav2vec2-base",
    freeze_encoder=True
).to(device)

print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters())}")

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = torch.nn.CrossEntropyLoss()

# Only train the classifier head (encoder is frozen)
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=1e-4
)

# =========================
# TRAINING LOOP
# =========================
epochs = 15
best_val_f1 = 0.0
patience = 3
patience_counter = 0

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(epochs):
    # ======================
    # TRAINING PHASE
    # ======================
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_preds = []
    train_targets = []

    for batch_idx, (x, y) in enumerate(train_loader):
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
        train_targets.extend(y.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}")

    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(train_targets, train_preds, average="weighted")
    train_loss /= len(train_loader)

    # ======================
    # VALIDATION PHASE
    # ======================
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0
    val_preds = []
    val_targets = []

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

            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(y.cpu().numpy())

    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(val_targets, val_preds, average="weighted")
    val_loss /= len(val_loader)

    # ======================
    # PRINT RESULTS
    # ======================
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1:   {val_f1:.4f}")

    # ======================
    # EARLY STOPPING & CHECKPOINT
    # ======================
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        # Save best model
        torch.save(
            model.state_dict(),
            "outputs/wav2vec2_model_best.pth"
        )
        print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping (patience {patience} reached)")
            break

# =========================
# FINAL SAVE
# =========================
torch.save(
    model.state_dict(),
    "outputs/wav2vec2_model_final.pth"
)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"Best validation F1: {best_val_f1:.4f}")
print("="*60)
