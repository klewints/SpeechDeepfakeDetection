
import os
import torch
import numpy as np

from torch.utils.data import (
    DataLoader,
    random_split,
    WeightedRandomSampler
)

from sklearn.metrics import f1_score

from src.dataset_loader import AudioDataset
from src.model import CNNModel


# =========================
# LOAD DATASET
# =========================

dataset = AudioDataset(
    "dataset/metadata.csv"
)

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
# GET TRAIN LABELS
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
# LOSS FUNCTION
# =========================

criterion = torch.nn.CrossEntropyLoss()

# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5
)

# =========================
# OUTPUT DIRECTORY
# =========================

os.makedirs("outputs", exist_ok=True)

# =========================
# BEST MODEL TRACKING
# =========================

best_val_f1 = 0.0

# =========================
# TRAINING LOOP
# =========================

epochs = 10

for epoch in range(epochs):

    # ======================
    # TRAINING
    # ======================

    model.train()

    train_loss = 0.0

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

    val_loss = 0.0

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

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    val_acc = 100 * val_correct / val_total

    # ======================
    # F1 SCORE
    # ======================

    val_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

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

    print(
        f"Validation F1 Score: {val_f1:.4f}"
    )

    # ======================
    # SAVE BEST MODEL
    # ======================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            model.state_dict(),
            "outputs/best_cnn_model.pth"
        )

        print(
            f"✓ Best CNN model saved! "
            f"F1 Score: {val_f1:.4f}"
        )

# =========================
# SAVE FINAL MODEL
# =========================

torch.save(
    model.state_dict(),
    "outputs/final_cnn_model.pth"
)

print("\nTraining complete.")

print(
    f"Best Validation F1 Score: "
    f"{best_val_f1:.4f}"
)

print("Final CNN model saved.")

