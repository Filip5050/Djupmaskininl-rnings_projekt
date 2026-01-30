import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from src.config import Config


class FraudPreprocessor:
    def __init__(self):
        # Only scale Time and Amount (V1-V28 are already PCA-scaled)
        self.scaler = RobustScaler()
        self.feature_cols_to_scale = [0, 29]  # Time (index 0) and Amount (index 29)
        
    def fit_transform(self, X, y):
        print("\nPreprocessing fraud data...")
        
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Scale only Time and Amount
        print("  Scaling features...")
        time_amount = X[:, self.feature_cols_to_scale]
        self.scaler.fit(time_amount)
        X[:, self.feature_cols_to_scale] = self.scaler.transform(time_amount)
        
        return X, y
    
    def transform(self, X):
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create a copy to avoid modifying original
        X_transformed = X.copy()
        
        # Scale only Time and Amount
        time_amount = X_transformed[:, self.feature_cols_to_scale]
        X_transformed[:, self.feature_cols_to_scale] = self.scaler.transform(time_amount)
        
        return X_transformed
    
    def save(self, path):
        preprocessor_data = {
            'scaler': self.scaler,
            'feature_cols_to_scale': self.feature_cols_to_scale
        }
        joblib.dump(preprocessor_data, path / 'preprocessor.pkl')
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path):
        print(f"Preprocessor loaded")
        preprocessor_data = joblib.load(path / 'preprocessor.pkl')
        
        preprocessor = FraudPreprocessor()
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.feature_cols_to_scale = preprocessor_data['feature_cols_to_scale']
        
        return preprocessor
