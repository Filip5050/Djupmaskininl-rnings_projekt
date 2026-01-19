import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Load test data and model
config_path = Path(r'C:\Code\Security_deep_learning')
X_test, y_test = joblib.load(config_path / 'models' / 'saved_models' / 'test_data.pkl')

from tensorflow import keras
model = keras.models.load_model(config_path / 'models' / 'saved_models' / 'fraud_model.h5')

# Get predictions
y_pred_prob = model.predict(X_test, verbose=0)

print("=" * 60)
print("PREDICTION PROBABILITY DISTRIBUTION")
print("=" * 60)
print(f"Min probability: {y_pred_prob.min():.6f}")
print(f"Max probability: {y_pred_prob.max():.6f}")
print(f"Mean probability: {y_pred_prob.mean():.6f}")
print(f"Median probability: {np.median(y_pred_prob):.6f}")
print()

# Test different thresholds
print("=" * 60)
print("RESULTS WITH DIFFERENT THRESHOLDS")
print("=" * 60)

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nThreshold: {threshold}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"  Recall: {recall*100:.1f}% (caught {tp}/{tp+fn} frauds)")
    print(f"  Precision: {precision*100:.1f}% (of {tp+fp} fraud predictions, {tp} were correct)")

print("\n" + "=" * 60)
print("RECOMMENDATION: Use threshold 0.2 or 0.3 for fraud detection")
print("=" * 60)
