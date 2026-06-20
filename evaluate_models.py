
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.dataset_loader import AudioDataset
from src.model import CNNModel
from src.resnet_model import ResNet18MelModel
from src.wav2vec_dataset import AudioRawDataset
from src.wav2vec_model import Wav2Vec2DeepfakeModel


# =====================================================
# EER + THRESHOLD FUNCTIONS
# =====================================================

def compute_eer(fpr, fnr):
    """
    Compute Equal Error Rate (EER).
    """

    abs_diffs = np.abs(fpr - fnr)

    min_idx = np.argmin(abs_diffs)

    eer = (fpr[min_idx] + fnr[min_idx]) / 2

    return eer, min_idx


def find_best_threshold(y_true, y_probs):
    """
    Find threshold corresponding to EER point.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_probs
    )

    fnr = 1 - tpr

    abs_diffs = np.abs(fpr - fnr)

    min_idx = np.argmin(abs_diffs)

    best_threshold = thresholds[min_idx]

    eer = (fpr[min_idx] + fnr[min_idx]) / 2

    return (
        best_threshold,
        eer,
        fpr[min_idx],
        fnr[min_idx],
        fpr,
        tpr
    )


# =====================================================
# MODEL EVALUATION
# =====================================================

def evaluate_model(
    model,
    test_loader,
    device,
    model_name,
    is_wav2vec=False
):

    print(f"\nEvaluating {model_name}...")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            fake_probs = probs[:, 1]

            all_probs.extend(
                fake_probs.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # =====================================================
    # FIND BEST THRESHOLD
    # =====================================================

    (
        best_threshold,
        eer,
        far,
        frr,
        fpr,
        tpr
    ) = find_best_threshold(
        all_labels,
        all_probs
    )

    # =====================================================
    # THRESHOLD-BASED PREDICTIONS
    # =====================================================

    all_preds = (
        all_probs > best_threshold
    ).astype(int)

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average='weighted'
    )

    roc_auc = roc_auc_score(
        all_labels,
        all_probs
    )

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")

    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'eer': eer,
        'far': far,
        'frr': frr,
        'threshold': best_threshold,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'fpr': fpr,
        'tpr': tpr
    }

    return metrics


# =====================================================
# ROC CURVES
# =====================================================

def plot_roc_curves(all_metrics, output_dir):

    plt.figure(figsize=(10, 8))

    for metrics in all_metrics:

        plt.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"{metrics['model']} (AUC={metrics['roc_auc']:.4f})",
            linewidth=2
        )

    plt.plot(
        [0, 1],
        [0, 1],
        'k--',
        label='Random'
    )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('ROC Curves')

    plt.legend()

    plt.grid(True)

    output_path = os.path.join(
        output_dir,
        'roc_curves.png'
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"\n✓ ROC curves saved: {output_path}")


# =====================================================
# CONFUSION MATRICES
# =====================================================

def plot_confusion_matrices(all_metrics, output_dir):

    fig, axes = plt.subplots(
        1,
        len(all_metrics),
        figsize=(5 * len(all_metrics), 4)
    )

    if len(all_metrics) == 1:
        axes = [axes]

    for idx, metrics in enumerate(all_metrics):

        cm = confusion_matrix(
            metrics['labels'],
            metrics['predictions']
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=False
        )

        axes[idx].set_title(
            metrics['model']
        )

        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

        axes[idx].set_xticklabels(
            ['Real', 'Fake']
        )

        axes[idx].set_yticklabels(
            ['Real', 'Fake']
        )

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        'confusion_matrices.png'
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"✓ Confusion matrices saved: {output_path}")


# =====================================================
# COMPARISON TABLE
# =====================================================

def create_comparison_table(
    all_metrics,
    output_dir
):

    data = []

    for metrics in all_metrics:

        data.append({
            'Model': metrics['model'],
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'EER': f"{metrics['eer']:.4f}",
            'FAR': f"{metrics['far']:.4f}",
            'FRR': f"{metrics['frr']:.4f}",
            'Threshold': f"{metrics['threshold']:.4f}"
        })

    df = pd.DataFrame(data)

    csv_path = os.path.join(
        output_dir,
        'comparison_metrics.csv'
    )

    df.to_csv(csv_path, index=False)

    print(f"\n✓ Metrics saved: {csv_path}")

    print("\n" + "=" * 120)
    print("FINAL MODEL COMPARISON")
    print("=" * 120)

    print(df.to_string(index=False))

    print("=" * 120)

    return df


# =====================================================
# MAIN
# =====================================================

print("=" * 60)
print("Advanced Model Evaluation")
print("=" * 60)

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# =====================================================
# LOAD DATASETS
# =====================================================

print("\nLoading datasets...")

dataset = AudioDataset("dataset/metadata.csv")

train_size = int(0.8 * len(dataset))

_, test_dataset = random_split(
    dataset,
    [train_size, len(dataset) - train_size]
)

test_loader_mel = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

dataset_wav2vec = AudioRawDataset(
    "dataset/metadata.csv",
    target_sr=16000
)

_, test_dataset_wav2vec = random_split(
    dataset_wav2vec,
    [train_size, len(dataset_wav2vec) - train_size]
)


def pad_collate_fn(batch):

    waveforms, labels = zip(*batch)

    max_len = max(
        w.shape[-1]
        for w in waveforms
    )

    padded_waveforms = []

    for w in waveforms:

        if w.shape[-1] < max_len:

            pad_len = max_len - w.shape[-1]

            w = torch.nn.functional.pad(
                w,
                (0, pad_len)
            )

        padded_waveforms.append(
            w.squeeze(0)
        )

    waveforms = torch.stack(
        padded_waveforms
    )

    labels = torch.stack(labels)

    return waveforms, labels


test_loader_wav2vec = DataLoader(
    test_dataset_wav2vec,
    batch_size=8,
    shuffle=False,
    collate_fn=pad_collate_fn
)

# =====================================================
# LOAD MODELS
# =====================================================

print("\nLoading models...")

all_metrics = []

# CNN
cnn_path = "outputs/best_cnn_model.pth"

if os.path.exists(cnn_path):

    cnn_model = CNNModel().to(device)

    cnn_model.load_state_dict(
        torch.load(
            cnn_path,
            map_location=device
        )
    )

    print("✓ CNN model loaded")

    cnn_metrics = evaluate_model(
        cnn_model,
        test_loader_mel,
        device,
        "CNN + Mel Spectrogram"
    )

    all_metrics.append(cnn_metrics)

# RESNET
resnet_path = "outputs/resnet/best_model.pth"

if os.path.exists(resnet_path):

    resnet_model = ResNet18MelModel().to(device)

    resnet_model.load_state_dict(
        torch.load(
            resnet_path,
            map_location=device
        )
    )

    print("✓ ResNet18 model loaded")

    resnet_metrics = evaluate_model(
        resnet_model,
        test_loader_mel,
        device,
        "ResNet18 + Mel Spectrogram"
    )

    all_metrics.append(resnet_metrics)

# WAV2VEC2
wav2vec_path = "outputs/wav2vec2/best_model.pth"

if os.path.exists(wav2vec_path):

    wav2vec_model = Wav2Vec2DeepfakeModel().to(device)

    wav2vec_model.load_state_dict(
        torch.load(
            wav2vec_path,
            map_location=device
        )
    )

    print("✓ Wav2Vec2 model loaded")

    wav2vec_metrics = evaluate_model(
        wav2vec_model,
        test_loader_wav2vec,
        device,
        "Wav2Vec2",
        is_wav2vec=True
    )

    all_metrics.append(wav2vec_metrics)

# =====================================================
# FINAL OUTPUTS
# =====================================================

if all_metrics:

    create_comparison_table(
        all_metrics,
        "outputs/metrics"
    )

    plot_roc_curves(
        all_metrics,
        "outputs/plots"
    )

    plot_confusion_matrices(
        all_metrics,
        "outputs/plots"
    )

    print("\nEvaluation complete!")

else:

    print("\nNo trained models found.")
