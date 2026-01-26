"""
Model training pipeline for fraud detection with PyTorch
"""
import time
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from src.config import Config
from src.data_loader import load_fraud_data, split_train_test, split_features_target
from src.preprocessor import FraudPreprocessor
from src.model_builder import build_fraud_classifier, save_model


def create_data_loader(X, y, batch_size, class_weights=None, shuffle=True):
    """Create PyTorch DataLoader with optional weighted sampling"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    if class_weights is not None and shuffle:
        # Weighted sampling for imbalanced data
        sample_weights = [class_weights[int(label)] for label in y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, (all_preds >= 0.5).astype(int), zero_division=0)
    recall = recall_score(all_labels, (all_preds >= 0.5).astype(int), zero_division=0)
    
    return total_loss / len(train_loader), accuracy, auc, precision, recall


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    accuracy = accuracy_score(all_labels, (all_preds >= 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, (all_preds >= 0.5).astype(int), zero_division=0)
    recall = recall_score(all_labels, (all_preds >= 0.5).astype(int), zero_division=0)
    
    return total_loss / len(val_loader), accuracy, auc, precision, recall


def full_training_pipeline():
    """Execute the complete fraud detection training pipeline"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - PYTORCH TRAINING PIPELINE")
    print("="*70)
    
    # Fix random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed: {Config.SEED} (for reproducibility)")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup directories
    Config.setup_directories()
    
    # 1. Load data
    print("\nSTEP 1: Loading Data")
    print("-"*60)
    df = load_fraud_data()
    
    # Split train/test
    train_df, test_df = split_train_test(df)
    X_train_full, y_train_full = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    
    # 2. Preprocess data
    print("\nSTEP 2: Preprocessing Data")
    print("-"*60)
    preprocessor = FraudPreprocessor()
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train_full, y_train_full)
    
    # Preprocess test data (just scaling)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Split train into train/val (from SMOTE-balanced data)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train_scaled,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=y_train_scaled
    )
    
    print(f"\nFinal dataset sizes:")
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test_scaled):,}")
    
    # 3. Create data loaders
    print("\nSTEP 3: Creating DataLoaders")
    print("-"*60)
    
    # Class weights for weighted sampling
    class_weights = {0: Config.NORMAL_WEIGHT, 1: Config.FRAUD_WEIGHT}
    
    train_loader = create_data_loader(
        X_train, y_train, 
        Config.BATCH_SIZE, 
        class_weights=class_weights,
        shuffle=True
    )
    val_loader = create_data_loader(
        X_val, y_val,
        Config.BATCH_SIZE,
        shuffle=False
    )
    
    print(f"DataLoaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 4. Build model
    print("\nSTEP 4: Building & Training Model")
    print("-"*60)
    model = build_fraud_classifier(X_train.shape[1])
    model = model.to(device)
    
    # Use Focal Loss for better handling of imbalanced data
    from src.model_builder import FocalLoss
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print("Using Focal Loss (alpha=0.25, gamma=2.0) for imbalanced data")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"   Architecture: {' -> '.join(map(str, Config.LAYER_SIZES))} -> 1")
    print(f"   Weighted sampling: Normal={Config.NORMAL_WEIGHT}, Fraud={Config.FRAUD_WEIGHT}")
    
    start_time = time.time()
    best_auc = 0
    patience_counter = 0
    
    history = {
        'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [],
        'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_precision': [], 'val_recall': []
    }
    
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_acc, train_auc, train_prec, train_rec = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_prec, val_rec = validate(
            model, val_loader, criterion, device
        )
        
        # Save history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['auc'].append(train_auc)
        history['precision'].append(train_prec)
        history['recall'].append(train_rec)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        
        # Print progress
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Early stopping & model checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            save_model(model, Config.MODELS_DIR / 'fraud_model.pt')
            print(f"  New best model saved! (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{Config.EARLY_STOPPING_PATIENCE})")
        
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    from src.model_builder import load_model
    model = load_model(Config.MODELS_DIR / 'fraud_model.pt', X_train.shape[1])
    model = model.to(device)
    
    # Final validation
    print("\nSTEP 5: Final Validation")
    print("-"*60)
    val_loss, val_acc, val_auc, val_prec, val_rec = validate(
        model, val_loader, criterion, device
    )
    
    print(f"   Val Loss:      {val_loss:.4f}")
    print(f"   Val Accuracy:  {val_acc:.4f}")
    print(f"   Val Precision: {val_prec:.4f}")
    print(f"   Val Recall:    {val_rec:.4f}")
    print(f"   Val AUC:       {val_auc:.4f}")
    
    # 6. Save everything
    print("\nSTEP 6: Saving Artifacts")
    print("-"*60)
    
    # Save preprocessor
    preprocessor.save(Config.MODELS_DIR)
    
    # Save test data for later evaluation (PREPROCESSED!)
    joblib.dump((X_test_scaled, y_test.values), Config.MODELS_DIR / 'test_data.pkl')
    print("Test data saved")
    
    # Save metadata
    metadata = {
        'model_type': 'PyTorch Fraud Detection Neural Network',
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': Config.FEATURE_COLS,
        'final_val_accuracy': val_acc,
        'final_val_auc': val_auc,
        'epochs_trained': len(history['loss']),
        'training_time': training_time,
        'device': str(device)
    }
    joblib.dump(metadata, Config.MODELS_DIR / 'model_metadata.pkl')
    print("Metadata saved")
    
    # Save training results for dashboard
    fraud_count_train = y_train.sum()
    normal_count_train = len(y_train) - fraud_count_train
    
    results = {
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'val_auc': float(val_auc),
        'val_precision': float(val_prec),
        'val_recall': float(val_rec),
        'history': {k: [float(x) for x in v] for k, v in history.items()},
        'class_distribution': {
            'Normal': int(normal_count_train),
            'Fraud': int(fraud_count_train)
        },
        'epochs_trained': len(history['loss']),
        'total_train_samples': len(X_train),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_time': f"{training_time:.1f}s"
    }
    
    joblib.dump(results, Config.MODELS_DIR / 'training_results.pkl')
    print("Training results saved")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Validation Metrics:")
    print(f"   Accuracy:  {val_acc:.2%}")
    print(f"   AUC:       {val_auc:.4f}")
    print(f"   Precision: {val_prec:.4f}")
    print(f"   Recall:    {val_rec:.4f}")
    print(f"Training Time: {training_time:.1f}s")
    print(f"Models saved to: {Config.MODELS_DIR}")
    print(f"\nNext steps:")
    print(f"   - python main.py evaluate  (evaluate on test data)")
    print(f"   - python main.py webapp    (launch dashboard)")


if __name__ == "__main__":
    full_training_pipeline()
