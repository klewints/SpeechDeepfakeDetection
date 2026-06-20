# Model Selection Refactoring: F1-Score to EER + ROC-AUC

## Overview
Refactored three training scripts to change best model selection from F1-score-based to EER (Equal Error Rate) and ROC-AUC based selection. This aligns with deepfake detection best practices where ranking quality matters more than threshold-dependent classification.

## Changes Summary

### Files Modified
1. **train.py** - CNN + Mel Spectrogram
2. **train_resnet.py** - ResNet18 + Mel Spectrogram
3. **train_wav2vec.py** - Wav2Vec2 Model

---

## Detailed Changes

### Common Changes Across All Scripts

#### 1. Added EER Computation Function
```python
def compute_eer(fpr, fnr):
    """
    Compute Equal Error Rate (EER) from FPR and FNR curves.
    EER is the point where FPR = FNR.
    """
    diff = np.abs(fpr - fnr)
    threshold_idx = np.argmin(diff)
    eer = (fpr[threshold_idx] + fnr[threshold_idx]) / 2
    return eer, threshold_idx
```

**Purpose**: Calculate EER by finding the intersection point of False Positive Rate and False Negative Rate curves.

#### 2. Updated Imports
- Added: `from sklearn.metrics import roc_curve, auc`
- These are needed for ROC curve computation

#### 3. Enhanced Validation Functions
**Updated validate_epoch() to compute:**
- Probability scores for positive class (via softmax)
- ROC curve (FPR, TPR, thresholds)
- ROC-AUC score
- EER value

**Return signature changed from:**
```python
return val_loss_avg, val_acc, val_f1
```
**To:**
```python
return val_loss_avg, val_acc, val_f1, eer, roc_auc
```

#### 4. Learning Rate Scheduler Mode Change
**Changed from:**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='max', ...)
```
**To:**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', ...)
```
**Reason**: EER is lower-is-better metric, opposite to F1-score.

#### 5. Best Model Selection Logic

**Old Logic (F1-based):**
```python
best_val_f1 = 0
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    torch.save(model.state_dict(), best_model_path)
```

**New Logic (EER + ROC-AUC based):**
```python
best_val_eer = float('inf')
best_val_roc_auc = 0

if val_eer < best_val_eer or (val_eer == best_val_eer and val_roc_auc > best_val_roc_auc):
    best_val_eer = val_eer
    best_val_roc_auc = val_roc_auc
    torch.save(model.state_dict(), best_model_path)
```

**Selection Criteria:**
- Primary: Lower EER (Equal Error Rate)
- Tie-breaker: Higher ROC-AUC if EER values are equal

#### 6. Enhanced Logging
**Added to each epoch:**
```
Val EER:    {val_eer:.4f} | ROC-AUC: {val_roc_auc:.4f}
✓ Best model saved (EER: {val_eer:.4f}, ROC-AUC: {val_roc_auc:.4f})
```

**Final output changed from:**
```
Training complete! Best F1: {best_val_f1:.4f}
```
**To:**
```
Training complete! Best EER: {best_val_eer:.4f}, Best ROC-AUC: {best_val_roc_auc:.4f}
```

---

## Script-Specific Changes

### train.py (CNN Model)
**Additional Changes:**
- Modified main training loop to collect predictions and probabilities
- Metrics that are still computed but not used for selection:
  - Accuracy: Still computed and logged
  - F1-score: Still computed and logged (for reporting only)
  
**Training loop structure:**
- Collects `train_preds` and `train_labels` during training
- Collects `val_probs` during validation (for ROC curve computation)
- Calls `compute_eer()` after computing ROC curve

### train_resnet.py (ResNet18 Model)
**Updates:**
- Updated `validate_epoch()` function signature
- Extracted probability scores: `probs = torch.softmax(outputs, dim=1); all_probs.extend(probs[:, 1].cpu().numpy())`
- Changed scheduler mode from 'max' to 'min'
- Updated best model selection logic with EER + ROC-AUC criteria

### train_wav2vec.py (Wav2Vec2 Model)
**Updates:**
- Same as ResNet18
- Includes early stopping logic (unchanged but now based on EER)
- Updated module docstring to mention EER and ROC-AUC based model selection

---

## Key Metrics Explanation

### Equal Error Rate (EER)
- **Definition**: The operating point where FPR = FNR
- **Why it matters**: For biometric/authentication tasks like deepfake detection, this represents the optimal operating point
- **Interpretation**: Lower EER = better model (ranges 0-1, where 0 is perfect)

### ROC-AUC (Area Under ROC Curve)
- **Definition**: Area under the Receiver Operating Characteristic curve
- **Why it matters**: Measures model's ability to distinguish between classes across all thresholds
- **Interpretation**: Higher ROC-AUC = better model (ranges 0-1, where 1 is perfect)

### Retained Metrics
- **F1-Score**: Still computed during validation but NOT used for model selection
- **Accuracy**: Still computed but NOT used for model selection
- **Loss**: Still computed but NOT used for model selection

---

## Benefits of This Refactoring

1. **Scientific Alignment**: EER is the standard metric for authentication/deepfake detection tasks
2. **Threshold-Independent**: EER doesn't assume a fixed 0.5 threshold; derives optimal threshold from ROC curve
3. **Tie-Breaking**: ROC-AUC provides secondary criterion when EER values are very similar
4. **Consistent Evaluation**: Same methodology applied across all three model architectures
5. **Backward Compatible**: Training and loss computation remain unchanged; only validation metrics for selection differ

---

## Validation Quality

All scripts have been:
- ✅ Syntax validated (Python compilation)
- ✅ Import verification (sklearn.metrics.roc_curve, auc)
- ✅ Consistent across all three training scripts
- ✅ Maintains existing training loop structure
- ✅ No changes to model architectures or dataset logic

---

## Testing Recommendations

1. Run each script with a small dataset subset to verify:
   - ROC curve computation works correctly
   - EER calculation produces valid values (0 ≤ EER ≤ 1)
   - Best model checkpoint is saved successfully
   - Logging output shows EER and ROC-AUC values

2. Monitor that:
   - EER improves (decreases) as training progresses
   - ROC-AUC generally improves (increases) as training progresses
   - Best model is being updated appropriately

3. Compare with previous runs:
   - Different model may be selected (EER vs F1 optimized models)
   - This is expected and correct; F1 and EER optimize for different objectives

---

## Code Quality
- All three scripts validated for syntax errors
- Consistent function signatures across implementations
- Clear documentation with inline comments
- Follows existing code style and conventions
