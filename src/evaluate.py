"""
Model evaluation on test data for fraud detection with PyTorch
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.config import Config
from src.preprocessor import FraudPreprocessor
from src.model_builder import load_model


def full_evaluation_pipeline():
    """Execute complete fraud detection evaluation pipeline"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PYTORCH EVALUATION")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 1. Load test data
    print("\nSTEP 1: Loading Test Data")
    print("-"*60)
    X_test, y_test = joblib.load(Config.MODELS_DIR / 'test_data.pkl')
    print(f"Loaded {len(X_test):,} test samples")
    fraud_count = y_test.sum()
    print(f"   Normal: {len(y_test) - fraud_count:,} | Fraud: {fraud_count:,}")
    
    # 2. Load model and preprocessor
    print("\nSTEP 2: Loading Model & Preprocessor")
    print("-"*60)
    preprocessor = FraudPreprocessor.load(Config.MODELS_DIR)
    model = load_model(Config.MODELS_DIR / 'fraud_model.pt', X_test.shape[1])
    model = model.to(device)
    model.eval()
    
    # 3. Make predictions
    print("\nSTEP 3: Making Predictions")
    print("-"*60)
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Predict in batches to avoid memory issues
    batch_size = 1024
    y_pred_proba = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            outputs = model(batch)
            y_pred_proba.extend(outputs.cpu().numpy())
    
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= 0.4).astype(int)
    
    print(f"Predictions complete (threshold: 0.4)")
    
    # 4. Calculate metrics
    print("\nSTEP 4: Calculating Metrics")
    print("-"*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC:      {auc:.4f}")
    
    # 5. Classification report
    print("\nClassification Report:")
    print("-"*60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Fraud'],
        zero_division=0
    ))
    
    # 6. Confusion matrix
    print("\nConfusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0,0]:,}")
    print(f"False Positives: {cm[0,1]:,}  (Normal flagged as Fraud)")
    print(f"False Negatives: {cm[1,0]:,}  (Fraud missed)")
    print(f"True Positives:  {cm[1,1]:,}")
    
    # 7. Visualizations
    print("\nSTEP 5: Creating Visualizations")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud'],
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    axes[1, 0].plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.4f}')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(alpha=0.3)
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='green')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
    axes[1, 1].axvline(x=0.4, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    save_path = Config.BASE_DIR / 'fraud_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualizations saved: {save_path}")
    
    # 8. Save results
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], output_dict=True)
    }
    joblib.dump(results, Config.MODELS_DIR / 'evaluation_results.pkl')
    print(f"Results saved")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC:      {auc:.4f}")
    print(f"   Fraud Detected: {cm[1,1]}/{fraud_count} ({cm[1,1]/fraud_count:.2%})")


if __name__ == "__main__":
    full_evaluation_pipeline()
