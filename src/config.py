"""
Configuration file for the Credit Card Fraud Detection System
"""
from pathlib import Path

class Config:
    """Central configuration for the fraud detection system"""
    

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'fraud_detection'
    MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    DATA_FILE = DATA_DIR / 'creditcard.csv'
    
    # Feature columns (V1-V28 are PCA components from original features)
    # Order matches dataset: Time, V1-V28, Amount
    FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    TARGET_COL = 'Class'  # 0 = Legitimate, 1 = Fraud
    
    # Model hyperparameters for binary classification
    LAYER_SIZES = [128, 64, 32, 16] 
    DROPOUT_RATES = [0.4, 0.35, 0.3, 0.25]
    ACTIVATION = 'relu'
    
    
    TEST_SIZE = 0.2
    SEED = 42  # Fixed seed for reproducible results
    BATCH_SIZE = 256  
    EPOCHS = 50  
    
    # Fraud detection specific
    THRESHOLD = 0.5  # Classification threshold
    FRAUD_WEIGHT = 900  # Increased from 700 for higher recall
    NORMAL_WEIGHT = 1  # Class weight for normal class
    EARLY_STOPPING_PATIENCE = 8  # Increased patience for better training
    LEARNING_RATE = 0.0005  # Lower LR for more stable convergence
    
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)