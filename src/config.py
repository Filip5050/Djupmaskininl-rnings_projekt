from pathlib import Path

class Config:
    
    BASE_DIR = Path(r'C:\Code\Security_deep_learning')
    DATA_DIR = BASE_DIR / 'data' / 'fraud_detection'
    MODELS_DIR = BASE_DIR / 'models' / 'saved_models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    DATA_FILE = DATA_DIR / 'creditcard.csv'
    
    # Feature columns (V1-V28 are PCA components from original features)
    # Order matches dataset: Time, V1-V28, Amount
    FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    TARGET_COL = 'Class'  # 0 = Legitimate, 1 = Fraud
    
    LAYER_SIZES = [128, 64, 32, 16] 
    DROPOUT_RATES = [0.4, 0.35, 0.3, 0.25]
    ACTIVATION = 'relu'
    
    
    TEST_SIZE = 0.2
    SEED = 42 
    BATCH_SIZE = 256  
    EPOCHS = 50  
    

    THRESHOLD = 0.5  
    FRAUD_WEIGHT = 900 
    NORMAL_WEIGHT = 1  
    EARLY_STOPPING_PATIENCE = 8
    LEARNING_RATE = 0.0005  
    
    
    @classmethod
    def setup_directories(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)