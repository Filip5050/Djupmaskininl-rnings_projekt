import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras

# Load test data and model
config_path = Path(r'C:\Code\Security_deep_learning')
X_test, y_test = joblib.load(config_path / 'models' / 'saved_models' / 'test_data.pkl')

model = keras.models.load_model(config_path / 'models' / 'saved_models' / 'fraud_model.h5')

# Get predictions
print("Predicting on test set...")
y_pred_prob = model.predict(X_test, verbose=0).flatten()

print("\n" + "="*70)
print("PREDICTION PROBABILITY ANALYSIS")
print("="*70)
print(f"Min probability: {y_pred_prob.min():.6f}")
print(f"Max probability: {y_pred_prob.max():.6f}")
print(f"Mean probability: {y_pred_prob.mean():.6f}")
print(f"Median probability: {np.median(y_pred_prob):.6f}")

# Look at fraud predictions specifically
fraud_indices = y_test == 1
fraud_probs = y_pred_prob[fraud_indices]
normal_probs = y_pred_prob[~fraud_indices]

print(f"\nFraud cases (n={len(fraud_probs)}):")
print(f"  Min: {fraud_probs.min():.6f}")
print(f"  Max: {fraud_probs.max():.6f}")
print(f"  Mean: {fraud_probs.mean():.6f}")
print(f"  Median: {np.median(fraud_probs):.6f}")

print(f"\nNormal cases (n={len(normal_probs)}):")
print(f"  Min: {normal_probs.min():.6f}")
print(f"  Max: {normal_probs.max():.6f}")
print(f"  Mean: {normal_probs.mean():.6f}")
print(f"  Median: {np.median(normal_probs):.6f}")

# AUC Score
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC Score: {auc:.4f}")

# Test different thresholds
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION")
print("="*70)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

best_f1 = 0
best_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_pred_prob > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
    
    print(f"\nThreshold: {threshold:.1f}")
    print(f"  TP={tp:5d} | FP={fp:5d} | FN={fn:3d} | TN={tn:5d}")
    print(f"  Recall:    {recall*100:5.1f}% (caught {tp}/{tp+fn} frauds)")
    print(f"  Precision: {precision*100:5.1f}% (of {tp+fp} fraud alerts, {tp} correct)")
    print(f"  F1-Score:  {f1:.4f}")

print("\n" + "="*70)
print(f"🎯 BEST THRESHOLD: {best_threshold:.1f} (F1-Score: {best_f1:.4f})")
print("="*70)
