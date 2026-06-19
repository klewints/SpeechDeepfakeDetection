import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import WeightedRandomSampler
import numpy as np
from src.dataset_loader import AudioDataset
from src.model import CNNModel

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

for epoch in range(epochs):

    # ======================
    # TRAINING
    # ======================

    model.train()

    train_loss = 0

    train_correct = 0

    train_total = 0

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

    train_acc = 100 * train_correct / train_total

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    val_correct = 0

    val_total = 0

    val_loss = 0

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

    val_acc = 100 * val_correct / val_total

    # ======================
    # PRINT RESULTS
    # ======================

    print(f"\nEpoch {epoch+1}/{epochs}")

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_acc:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_acc:.2f}%"
    )

# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "outputs/model.pth"
)

print("Model saved.")