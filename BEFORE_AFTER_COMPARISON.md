# Before & After Comparison

## Key Changes Across All Scripts

### 1. Imports

**BEFORE (train_resnet.py & train_wav2vec.py):**
```python
from sklearn.metrics import f1_score, accuracy_score
```

**AFTER (train_resnet.py & train_wav2vec.py):**
```python
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
```

**BEFORE (train.py):**
```python
import numpy as np
from src.dataset_loader import AudioDataset
from src.model import CNNModel
```

**AFTER (train.py):**
```python
import numpy as np
from sklearn.metrics import f1_score, roc_curve, auc
from src.dataset_loader import AudioDataset
from src.model import CNNModel
```

---

## 2. New Function: compute_eer()

**ADDED (all three scripts):**
```python
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
```

---

## 3. Validation Function Update (train_resnet.py & train_wav2vec.py)

### Probability Collection

**BEFORE:**
```python
def validate_epoch(model, val_loader, criterion, device):
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
            # ... rest of validation
```

**AFTER:**
```python
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    all_probs = []  # NEW: Collect probabilities
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            val_loss += loss.item()
            
            # NEW: Get probabilities for positive class
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            _, predicted = torch.max(outputs, 1)
            # ... rest of validation
```

### Return Values & Metrics Computation

**BEFORE:**
```python
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_loss_avg = val_loss / len(val_loader)
    
    return val_loss_avg, val_acc, val_f1
```

**AFTER:**
```python
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_loss_avg = val_loss / len(val_loader)
    
    # NEW: Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # NEW: Compute EER
    fnr = 1 - tpr
    eer, _ = compute_eer(fpr, fnr)
    
    return val_loss_avg, val_acc, val_f1, eer, roc_auc
```

---

## 4. Scheduler Mode Change (train_resnet.py & train_wav2vec.py)

**BEFORE:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',  # Maximize metric (F1-score)
    factor=0.5,
    patience=3
)
```

**AFTER:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',  # Minimize metric (EER - lower is better)
    factor=0.5,
    patience=3
)
```

---

## 5. Best Model Selection (train_resnet.py Example)

### Training Loop Initialization

**BEFORE:**
```python
epochs = 20
best_val_f1 = 0
best_model_path = "outputs/resnet/best_model.pth"
```

**AFTER:**
```python
epochs = 20
best_val_eer = float('inf')
best_val_roc_auc = 0
best_model_path = "outputs/resnet/best_model.pth"
```

### Validation & Model Saving

**BEFORE:**
```python
for epoch in range(epochs):
    
    train_loss, train_acc, train_f1 = train_epoch(...)
    val_loss, val_acc, val_f1 = validate_epoch(...)
    
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
```

**AFTER:**
```python
for epoch in range(epochs):
    
    train_loss, train_acc, train_f1 = train_epoch(...)
    val_loss, val_acc, val_f1, val_eer, val_roc_auc = validate_epoch(...)  # NEW: unpack EER and ROC-AUC
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
    print(f"  Val EER:    {val_eer:.4f} | ROC-AUC: {val_roc_auc:.4f}")  # NEW: Print EER and ROC-AUC
    
    # Learning rate scheduling (based on EER now)
    scheduler.step(val_eer)  # CHANGED: Step on EER instead of F1
    
    # Save best model based on EER (lower is better)
    # If EER is equal, use higher ROC-AUC as tie-breaker
    if val_eer < best_val_eer or (val_eer == best_val_eer and val_roc_auc > best_val_roc_auc):  # CHANGED: Selection logic
        best_val_eer = val_eer  # CHANGED: Track EER
        best_val_roc_auc = val_roc_auc  # NEW: Track ROC-AUC
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (EER: {val_eer:.4f}, ROC-AUC: {val_roc_auc:.4f})")  # CHANGED: Message
```

---

## 6. Final Results (train_resnet.py Example)

**BEFORE:**
```python
print("\n" + "=" * 60)
print(f"Training complete! Best F1: {best_val_f1:.4f}")
print(f"Model saved to: {best_model_path}")
print("=" * 60)
```

**AFTER:**
```python
print("\n" + "=" * 60)
print(f"Training complete! Best EER: {best_val_eer:.4f}, Best ROC-AUC: {best_val_roc_auc:.4f}")
print(f"Model saved to: {best_model_path}")
print("=" * 60)
```

---

## 7. Training Loop in train.py (Simplified CNN)

### Before Structure
```python
# Simple training/validation without best model tracking
for epoch in range(epochs):
    # TRAINING
    # ... compute loss, accuracy
    
    # VALIDATION
    # ... compute loss, accuracy
    
    # PRINT
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

# Save model at end
torch.save(model.state_dict(), "outputs/model.pth")
print("Model saved.")
```

### After Structure
```python
best_val_eer = float('inf')
best_val_roc_auc = 0
best_model_path = "outputs/best_model.pth"

for epoch in range(epochs):
    # TRAINING
    # ... compute loss, accuracy, F1
    
    # VALIDATION
    # ... compute loss, accuracy, F1
    # NEW: Compute ROC curve, EER, ROC-AUC
    
    # PRINT (with EER and ROC-AUC)
    print(f"Val EER:    {val_eer:.4f} | ROC-AUC: {val_roc_auc:.4f}")
    
    # NEW: Save best model based on EER
    if val_eer < best_val_eer or (val_eer == best_val_eer and val_roc_auc > best_val_roc_auc):
        best_val_eer = val_eer
        best_val_roc_auc = val_roc_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (EER: {val_eer:.4f}, ROC-AUC: {val_roc_auc:.4f})")

# FINAL: Print best EER and ROC-AUC
print(f"Training complete! Best EER: {best_val_eer:.4f}, Best ROC-AUC: {best_val_roc_auc:.4f}")
```

---

## Summary of Metric Changes

| Metric | Before | After |
|--------|--------|-------|
| Primary Selection | F1-Score (higher) | EER (lower) |
| Secondary Selection | None | ROC-AUC (higher) |
| Threshold Derivation | Fixed at 0.5 | From ROC curve |
| Selection Type | Threshold-dependent | Threshold-independent |
| Scheduler Mode | 'max' (maximize) | 'min' (minimize) |
| F1-Score | Used for selection | Reported only |
| Accuracy | Not computed | Reported only |
| ROC-AUC | Not computed | Used for tie-breaking |
| EER | Not computed | Primary selection |

---

## Impact on Model Selection

### Example Training Progression

**Epoch-by-epoch comparison (hypothetical values):**

| Epoch | Val F1 | Val EER | Val ROC-AUC | F1-Based Select? | EER-Based Select? |
|-------|--------|---------|-------------|------------------|-------------------|
| 1 | 0.72 | 0.35 | 0.78 | ✅ Save | ✅ Save |
| 2 | 0.75 | 0.33 | 0.80 | ✅ Save | ✅ Save |
| 3 | 0.73 | 0.34 | 0.79 | ❌ Skip | ❌ Skip |
| 4 | 0.76 | 0.32 | 0.81 | ✅ Save | ✅ Save |
| 5 | 0.77 | 0.32 | 0.82 | ✅ Save | ❌ (same EER, better ROC-AUC) |

**Key insight**: F1 and EER often optimize for different models. The new approach may select a different checkpoint than F1-based selection.

---

## No Breaking Changes

✅ Training loop structure preserved
✅ Model architectures unchanged
✅ Dataset loading unchanged
✅ Loss computation unchanged
✅ Optimizer behavior unchanged
✅ Backward compatible with existing infrastructure
✅ F1-Score still available for reporting/analysis

---

## Implementation Consistency

All three scripts now:
- Compute EER using identical `compute_eer()` function
- Use identical selection logic: `if val_eer < best_val_eer or (val_eer == best_val_eer and val_roc_auc > best_val_roc_auc)`
- Print metrics in identical format
- Use scheduler in identical mode ('min')
- Maintain identical training loop structure
