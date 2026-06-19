"""
Advanced evaluation pipeline comparing all three models.

Evaluates CNN, ResNet18, and Wav2Vec2 models and generates:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Anti-spoofing metrics (EER, FAR, FRR)
- Visualization plots
- Comparison table
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.dataset_loader import AudioDataset
from src.model import CNNModel
from src.resnet_model import ResNet18MelModel
from src.wav2vec_dataset import AudioRawDataset
from src.wav2vec_model import Wav2Vec2DeepfakeModel


def compute_eer(fpr, fnr):
    """
    Compute Equal Error Rate (EER) from FPR and FNR curves.
    
    EER is the point where False Positive Rate equals False Negative Rate.
    """
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    eer = max(fpr[min_idx], fnr[min_idx])
    return eer


def evaluate_model(model, test_loader, device, model_name, is_wav2vec=False):
    """
    Evaluate a single model.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: torch device
        model_name: Name of model for display
        is_wav2vec: Whether this is a Wav2Vec2 model (handles different input)
    
    Returns:
        Dictionary with all metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if is_wav2vec:
                x = x.to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # =========================
    # CLASSIFICATION METRICS
    # =========================
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # =========================
    # ANTI-SPOOFING METRICS
    # =========================
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer = compute_eer(fpr, fnr)
    
    # FAR and FRR at EER point
    abs_diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diffs)
    far_at_eer = fpr[min_idx]
    frr_at_eer = fnr[min_idx]
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'eer': eer,
        'far': far_at_eer,
        'frr': frr_at_eer,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'fpr': fpr,
        'tpr': tpr
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  EER: {eer:.4f}")
    
    return metrics


def plot_roc_curves(all_metrics, output_dir):
    """
    Plot ROC curves for all models.
    """
    plt.figure(figsize=(10, 8))
    
    for metrics in all_metrics:
        plt.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"{metrics['model']} (AUC={metrics['roc_auc']:.4f})",
            linewidth=2
        )
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curves saved: {output_path}")
    plt.close()


def plot_confusion_matrices(all_metrics, output_dir):
    """
    Plot confusion matrices for all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, metrics in enumerate(all_metrics):
        cm = confusion_matrix(metrics['labels'], metrics['predictions'])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=False
        )
        
        axes[idx].set_title(metrics['model'], fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_xticklabels(['Real', 'Fake'])
        axes[idx].set_yticklabels(['Real', 'Fake'])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {output_path}")
    plt.close()


def create_comparison_table(all_metrics, output_dir):
    """
    Create and save comparison table.
    """
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
            'FRR': f"{metrics['frr']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'comparison_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Metrics saved: {csv_path}")
    
    # Print table
    print("\n" + "=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    return df


# =========================
# SETUP
# =========================

print("=" * 60)
print("Advanced Model Evaluation")
print("=" * 60)

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# LOAD TEST DATASET (MEL SPECTROGRAM)
# =========================

print("\nLoading test dataset...")
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

# =========================
# LOAD TEST DATASET (RAW AUDIO)
# =========================

dataset_wav2vec = AudioRawDataset("dataset/metadata.csv", target_sr=16000)
_, test_dataset_wav2vec = random_split(
    dataset_wav2vec,
    [train_size, len(dataset_wav2vec) - train_size]
)

def pad_collate_fn(batch):
    waveforms, labels = zip(*batch)
    max_len = max(w.shape[-1] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.shape[-1] < max_len:
            pad_len = max_len - w.shape[-1]
            w = torch.nn.functional.pad(w, (0, pad_len))
        padded_waveforms.append(w.squeeze(0))
    waveforms = torch.stack(padded_waveforms)
    labels = torch.stack(labels)
    return waveforms, labels

test_loader_wav2vec = DataLoader(
    test_dataset_wav2vec,
    batch_size=8,
    shuffle=False,
    collate_fn=pad_collate_fn
)

# =========================
# LOAD MODELS
# =========================

print("\nLoading models...")

# CNN Model
cnn_model = CNNModel().to(device)
cnn_path = "outputs/model.pth"
if os.path.exists(cnn_path):
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    print("✓ CNN model loaded")
else:
    print("⚠ CNN model not found, skipping...")

# ResNet Model
resnet_model = ResNet18MelModel().to(device)
resnet_path = "outputs/resnet/best_model.pth"
if os.path.exists(resnet_path):
    resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
    print("✓ ResNet18 model loaded")
else:
    print("⚠ ResNet18 model not found, skipping...")

# Wav2Vec2 Model
wav2vec_model = Wav2Vec2DeepfakeModel().to(device)
wav2vec_path = "outputs/wav2vec2/best_model.pth"
if os.path.exists(wav2vec_path):
    wav2vec_model.load_state_dict(torch.load(wav2vec_path, map_location=device))
    print("✓ Wav2Vec2 model loaded")
else:
    print("⚠ Wav2Vec2 model not found, skipping...")

# =========================
# EVALUATE MODELS
# =========================

all_metrics = []

if os.path.exists(cnn_path):
    cnn_metrics = evaluate_model(
        cnn_model, test_loader_mel, device, 
        "CNN + Mel Spectrogram", is_wav2vec=False
    )
    all_metrics.append(cnn_metrics)

if os.path.exists(resnet_path):
    resnet_metrics = evaluate_model(
        resnet_model, test_loader_mel, device, 
        "ResNet18 + Mel Spectrogram", is_wav2vec=False
    )
    all_metrics.append(resnet_metrics)

if os.path.exists(wav2vec_path):
    wav2vec_metrics = evaluate_model(
        wav2vec_model, test_loader_wav2vec, device, 
        "Wav2Vec2", is_wav2vec=True
    )
    all_metrics.append(wav2vec_metrics)

# =========================
# SAVE COMPARISON TABLE
# =========================

if all_metrics:
    comparison_df = create_comparison_table(all_metrics, "outputs/metrics")
    
    # =========================
    # GENERATE VISUALIZATIONS
    # =========================
    
    plot_roc_curves(all_metrics, "outputs/plots")
    plot_confusion_matrices(all_metrics, "outputs/plots")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
else:
    print("\n⚠ No models found to evaluate. Please train models first.")
