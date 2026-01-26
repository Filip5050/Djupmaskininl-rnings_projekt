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


def calculate_feature_importance(model, X_test, y_test, feature_names, device, n_repeats=10):
    """
    Calculate feature importance using permutation importance
    
    Args:
        model: Trained PyTorch model
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        feature_names: List of feature names
        device: torch device
        n_repeats: Number of permutation repeats
        
    Returns:
        DataFrame with feature importances sorted by importance
    """
    print("\nCalculating feature importance (this may take a minute)...")
    
    # Get baseline AUC
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_baseline = model(X_tensor).cpu().numpy().flatten()
        baseline_auc = roc_auc_score(y_test, y_pred_baseline)
    
    importances = []
    
    for feature_idx, feature_name in enumerate(feature_names):
        importance_scores = []
        
        for _ in range(n_repeats):
            # Create a copy and shuffle this feature
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Get new AUC
            with torch.no_grad():
                X_tensor_perm = torch.FloatTensor(X_permuted).to(device)
                y_pred_perm = model(X_tensor_perm).cpu().numpy().flatten()
                permuted_auc = roc_auc_score(y_test, y_pred_perm)
            
            # Importance = drop in AUC
            importance_scores.append(baseline_auc - permuted_auc)
        
        importances.append({
            'feature': feature_name,
            'importance': np.mean(importance_scores),
            'std': np.std(importance_scores)
        })
    
    # Create DataFrame
    importance_df = pd.DataFrame(importances).sort_values('importance', ascending=False)
    
    return importance_df


def full_evaluation_pipeline():
    """Execute complete fraud detection evaluation pipeline"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PYTORCH EVALUATION")
    print("="*70)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
  
    print("\nSTEP 1: Loading Test Data")
    print("-"*60)
    X_test, y_test = joblib.load(Config.MODELS_DIR / 'test_data.pkl')
    print(f"Loaded {len(X_test):,} test samples")
    fraud_count = y_test.sum()
    print(f"   Normal: {len(y_test) - fraud_count:,} | Fraud: {fraud_count:,}")

    print("\nSTEP 2: Loading Model & Preprocessor")
    print("-"*60)
    preprocessor = FraudPreprocessor.load(Config.MODELS_DIR)
    model = load_model(Config.MODELS_DIR / 'fraud_model.pt', X_test.shape[1])
    model = model.to(device)
    model.eval()
    
    print("\nSTEP 3: Making Predictions")
    print("-"*60)
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    batch_size = 1024
    y_pred_proba = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            outputs = model(batch)
            y_pred_proba.extend(outputs.cpu().numpy())
    
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print(f"Predictions complete (threshold: 0.5)")
    
    print("\nSTEP 4: Calculating Metrics")
    print("-"*60)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nTest Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC:      {auc:.4f}")
    
    print("\nClassification Report:")
    print("-"*60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Fraud'],
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    print("-"*60)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"True Negatives:  {tn:,}")
    print(f"False Positives: {fp:,}  (Normal flagged as Fraud)")
    print(f"False Negatives: {fn:,}  (Fraud missed)")
    print(f"True Positives:  {tp:,}")
    
    print("\nAdvanced Fraud Detection Metrics:")
    print("-"*60)
    
    overall_fraud_rate = fraud_count / len(y_test)
    
    flagged_count = tp + fp
    fraud_in_flagged = tp
    fraud_rate_in_flagged = fraud_in_flagged / flagged_count if flagged_count > 0 else 0
    
    # Lift: How much better than random selection
    lift = fraud_rate_in_flagged / overall_fraud_rate if overall_fraud_rate > 0 else 0
    
    # Fraud Positive Rate (Precision at cutoff)
    fraud_positive_rate = tp / flagged_count if flagged_count > 0 else 0
    
    # Flagging rate (what % of transactions are flagged)
    flagging_rate = flagged_count / len(y_test)
    
    print(f"Overall Fraud Rate:        {overall_fraud_rate*100:.3f}%")
    print(f"Flagging Rate:             {flagging_rate*100:.2f}% (Top {flagging_rate*100:.1f}% flagged for review)")
    print(f"Fraud Rate in Flagged:     {fraud_rate_in_flagged*100:.2f}%")
    print(f"Lift:                      {lift:.1f}x (Model is {lift:.1f}x better than random)")
    print(f"Fraud Positive Rate:       {fraud_positive_rate*100:.2f}% ({tp}/{flagged_count} flagged are fraud)")
    print(f"Fraud Capture Rate:        {tp/fraud_count*100:.2f}% ({tp}/{fraud_count} frauds caught)")
    
    print("\nInterpretation (Industry Standards):")
    print("-"*60)
    if lift > 15:
        print(f"✓ Lift {lift:.1f}x is EXCELLENT (>8 is considered strong)")
    elif lift > 8:
        print(f"✓ Lift {lift:.1f}x is STRONG (>8 is considered strong)")
    else:
        print(f"⚠ Lift {lift:.1f}x could be improved (target >8)")
    
    if flagging_rate <= 0.05:
        print(f"✓ Flagging rate {flagging_rate*100:.1f}% matches 'Top 5%' industry approach")
    elif flagging_rate <= 0.02:
        print(f"✓ Flagging rate {flagging_rate*100:.1f}% is conservative (Top 2%)")
    else:
        print(f"⚠ Flagging rate {flagging_rate*100:.1f}% is high (consider raising threshold)")
    
    # 7. Visualizations
    print("\nSTEP 5: Creating Visualizations")
    print("-"*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
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
    axes[1, 1].axvline(x=Config.THRESHOLD, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    save_path = Config.BASE_DIR / 'fraud_evaluation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualizations saved: {save_path}")
    plt.close()
    
    print("\nSTEP 6: Feature Importance Analysis")
    print("-"*60)
    
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    importance_df = calculate_feature_importance(
        model, X_test, y_test, feature_names, device, n_repeats=5
    )
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(15)
    
    ax.barh(range(len(top_features)), top_features['importance'], 
            xerr=top_features['std'], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance (Change in AUC when permuted)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Top 15 Feature Importance (Permutation)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    importance_path = Config.BASE_DIR / 'feature_importance.png'
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved: {importance_path}")
    plt.close()
    
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], output_dict=True),
        'feature_importance': importance_df.to_dict(),
        'advanced_metrics': {
            'lift': float(lift),
            'fraud_positive_rate': float(fraud_positive_rate),
            'flagging_rate': float(flagging_rate),
            'overall_fraud_rate': float(overall_fraud_rate),
            'fraud_capture_rate': float(tp/fraud_count)
        }
    }
    joblib.dump(results, Config.MODELS_DIR / 'evaluation_results.pkl')
    print(f"Results saved")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Metrics:")
    print(f"   Accuracy:               {accuracy:.4f}")
    print(f"   AUC:                    {auc:.4f}")
    print(f"   Fraud Detected:         {tp}/{fraud_count} ({tp/fraud_count:.2%})")
    print(f"   Lift:                   {lift:.1f}x")
    print(f"   Fraud Positive Rate:    {fraud_positive_rate*100:.2f}%")
    print(f"\nMost Important Feature: {importance_df.iloc[0]['feature']} "
          f"(importance: {importance_df.iloc[0]['importance']:.4f})")


if __name__ == "__main__":
    full_evaluation_pipeline()
