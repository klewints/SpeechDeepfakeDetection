import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

from src.dataset_loader import AudioDataset
from src.wav2vec_dataset import Wav2Vec2Dataset
from src.model import CNNModel
from src.wav2vec_model import Wav2Vec2Classifier


# =========================
# UTILITY FUNCTIONS
# =========================

def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER).
    
    EER is the point where FAR (False Alarm Rate) = FRR (False Rejection Rate).
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Confidence scores (probabilities) for positive class
    
    Returns:
        eer: Equal Error Rate value
        threshold: EER threshold
        far_list: False Alarm Rates
        frr_list: False Rejection Rates
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # FRR = 1 - TPR
    frr = 1 - tpr
    far = fpr
    
    # Find the threshold where FAR ≈ FRR
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    
    eer = (far[eer_idx] + frr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold, far, frr, fpr, tpr, thresholds


def evaluate_model(model, dataloader, device, model_type="CNN"):
    """
    Evaluate model and collect predictions and scores.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: torch device
        model_type: "CNN" or "Wav2Vec2" (for logging)
    
    Returns:
        Dictionary with predictions, targets, and scores
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_scores = []  # Confidence scores for positive class
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Get scores (softmax probabilities for positive class)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            scores = probs[:, 1].cpu().numpy()  # Probability of fake class
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_scores.extend(scores)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    [{model_type}] Batch {batch_idx+1}/{len(dataloader)}")
    
    return {
        "predictions": np.array(all_preds),
        "targets": np.array(all_targets),
        "scores": np.array(all_scores)
    }


def compute_metrics(y_true, y_pred, y_scores):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Confidence scores
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
    }
    
    # Compute EER
    eer, eer_threshold, far, frr, fpr, tpr, thresholds = compute_eer(y_true, y_scores)
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold
    metrics["far"] = far
    metrics["frr"] = frr
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["thresholds"] = thresholds
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    return metrics


def plot_confusion_matrix(cm, model_name, save_path="outputs"):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    filename = os.path.join(save_path, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()


def plot_roc_curves(results_cnn, results_wav2vec, save_path="outputs"):
    """Plot and save ROC curves for both models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # CNN ROC curve
    ax.plot(
        results_cnn["metrics"]["fpr"],
        results_cnn["metrics"]["tpr"],
        label=f"CNN + Mel (AUC = {results_cnn['metrics']['roc_auc']:.3f})",
        linewidth=2,
        color="blue"
    )
    
    # Wav2Vec2 ROC curve
    ax.plot(
        results_wav2vec["metrics"]["fpr"],
        results_wav2vec["metrics"]["tpr"],
        label=f"Wav2Vec2 (AUC = {results_wav2vec['metrics']['roc_auc']:.3f})",
        linewidth=2,
        color="orange"
    )
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    filename = os.path.join(save_path, "roc_curves_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()


def print_comparison_table(results_cnn, results_wav2vec):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print(" "*20 + "MODEL COMPARISON - EVALUATION METRICS")
    print("="*80)
    
    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc", "eer"]
    
    print(f"\n{'Metric':<15} {'CNN + Mel':<25} {'Wav2Vec2':<25} {'Difference':<15}")
    print("-"*80)
    
    for metric in metrics_list:
        cnn_val = results_cnn["metrics"][metric]
        wav2vec_val = results_wav2vec["metrics"][metric]
        diff = wav2vec_val - cnn_val
        
        # Format percentage
        if metric in ["roc_auc", "eer"]:
            print(f"{metric:<15} {cnn_val:<25.4f} {wav2vec_val:<25.4f} {diff:+.4f}")
        else:
            cnn_pct = cnn_val * 100
            wav2vec_pct = wav2vec_val * 100
            diff_pct = diff * 100
            print(f"{metric:<15} {cnn_pct:<24.2f}% {wav2vec_pct:<24.2f}% {diff_pct:+.2f}%")
    
    print("="*80)
    
    # Confusion matrices
    print("\nCONFUSION MATRICES:")
    print("-"*80)
    
    cm_cnn = results_cnn["metrics"]["confusion_matrix"]
    cm_wav2vec = results_wav2vec["metrics"]["confusion_matrix"]
    
    print("\nCNN + Mel:")
    print(f"  True Negatives (Real):   {cm_cnn[0, 0]}")
    print(f"  False Positives (Fake):  {cm_cnn[0, 1]}")
    print(f"  False Negatives (Real):  {cm_cnn[1, 0]}")
    print(f"  True Positives (Fake):   {cm_cnn[1, 1]}")
    
    print("\nWav2Vec2:")
    print(f"  True Negatives (Real):   {cm_wav2vec[0, 0]}")
    print(f"  False Positives (Fake):  {cm_wav2vec[0, 1]}")
    print(f"  False Negatives (Real):  {cm_wav2vec[1, 0]}")
    print(f"  True Positives (Fake):   {cm_wav2vec[1, 1]}")
    
    print("="*80)


# =========================
# SETUP
# =========================
os.makedirs("outputs", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# =========================
# LOAD TEST DATA
# =========================
print("Loading test datasets...")

# CNN uses Mel Spectrograms
dataset_cnn = AudioDataset("dataset/metadata.csv")
test_loader_cnn = DataLoader(dataset_cnn, batch_size=16, shuffle=False)

# Wav2Vec2 uses raw waveforms
dataset_wav2vec = Wav2Vec2Dataset("dataset/metadata.csv")
test_loader_wav2vec = DataLoader(dataset_wav2vec, batch_size=8, shuffle=False)

print(f"Total test samples: {len(dataset_cnn)}\n")

# =========================
# LOAD CNN MODEL
# =========================
print("Loading CNN model...")
cnn_model = CNNModel().to(device)
if os.path.exists("outputs/model.pth"):
    cnn_model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    print("✓ CNN model loaded from outputs/model.pth")
else:
    print("⚠ CNN model not found. Make sure to run train.py first.")
    print("  Looking for: outputs/model.pth")

# =========================
# LOAD WAV2VEC2 MODEL
# =========================
print("\nLoading Wav2Vec2 model...")
wav2vec_model = Wav2Vec2Classifier().to(device)
if os.path.exists("outputs/wav2vec2_model_best.pth"):
    wav2vec_model.load_state_dict(
        torch.load("outputs/wav2vec2_model_best.pth", map_location=device)
    )
    print("✓ Wav2Vec2 model loaded from outputs/wav2vec2_model_best.pth")
elif os.path.exists("outputs/wav2vec2_model_final.pth"):
    wav2vec_model.load_state_dict(
        torch.load("outputs/wav2vec2_model_final.pth", map_location=device)
    )
    print("✓ Wav2Vec2 model loaded from outputs/wav2vec2_model_final.pth")
else:
    print("⚠ Wav2Vec2 model not found. Make sure to run train_wav2vec.py first.")
    print("  Looking for: outputs/wav2vec2_model_best.pth or outputs/wav2vec2_model_final.pth")

# =========================
# EVALUATE CNN MODEL
# =========================
print("\n" + "="*60)
print("EVALUATING CNN MODEL (Mel Spectrogram)")
print("="*60)

results_cnn = evaluate_model(cnn_model, test_loader_cnn, device, "CNN")
results_cnn["metrics"] = compute_metrics(
    results_cnn["targets"],
    results_cnn["predictions"],
    results_cnn["scores"]
)

print("✓ CNN evaluation complete")

# =========================
# EVALUATE WAV2VEC2 MODEL
# =========================
print("\n" + "="*60)
print("EVALUATING WAV2VEC2 MODEL")
print("="*60)

results_wav2vec = evaluate_model(wav2vec_model, test_loader_wav2vec, device, "Wav2Vec2")
results_wav2vec["metrics"] = compute_metrics(
    results_wav2vec["targets"],
    results_wav2vec["predictions"],
    results_wav2vec["scores"]
)

print("✓ Wav2Vec2 evaluation complete")

# =========================
# GENERATE VISUALIZATIONS
# =========================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

print("\nCreating confusion matrix plots...")
plot_confusion_matrix(results_cnn["metrics"]["confusion_matrix"], "CNN_Mel")
plot_confusion_matrix(results_wav2vec["metrics"]["confusion_matrix"], "Wav2Vec2")

print("\nCreating ROC curve comparison...")
plot_roc_curves(results_cnn, results_wav2vec)

# =========================
# PRINT RESULTS
# =========================
print_comparison_table(results_cnn, results_wav2vec)

# =========================
# SAVE RESULTS TO CSV
# =========================
print("\nSaving detailed results to CSV...")

results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "EER"],
    "CNN + Mel": [
        f"{results_cnn['metrics']['accuracy']*100:.2f}%",
        f"{results_cnn['metrics']['precision']*100:.2f}%",
        f"{results_cnn['metrics']['recall']*100:.2f}%",
        f"{results_cnn['metrics']['f1']*100:.2f}%",
        f"{results_cnn['metrics']['roc_auc']:.4f}",
        f"{results_cnn['metrics']['eer']:.4f}",
    ],
    "Wav2Vec2": [
        f"{results_wav2vec['metrics']['accuracy']*100:.2f}%",
        f"{results_wav2vec['metrics']['precision']*100:.2f}%",
        f"{results_wav2vec['metrics']['recall']*100:.2f}%",
        f"{results_wav2vec['metrics']['f1']*100:.2f}%",
        f"{results_wav2vec['metrics']['roc_auc']:.4f}",
        f"{results_wav2vec['metrics']['eer']:.4f}",
    ]
})

csv_path = "outputs/evaluation_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"✓ Results saved to {csv_path}")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - outputs/confusion_matrix_CNN_Mel.png")
print("  - outputs/confusion_matrix_Wav2Vec2.png")
print("  - outputs/roc_curves_comparison.png")
print("  - outputs/evaluation_results.csv")
print("="*80)
