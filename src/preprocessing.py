# filepath: c:\Code\Security_deep_learning\src\preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.le_target = LabelEncoder()
        
    def load_data(self, file_path, column_names):
        """Load data from file"""
        return pd.read_csv(file_path, header=None, names=column_names)
    
    def encode_features(self, X, fit=True):
        """Encode categorical features"""
        X_encoded = X.copy()
        
        for col in self.config.CATEGORICAL_COLS:
            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unknown values
                X_encoded[col] = X_encoded[col].map(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_encoded[col] = le.transform(X_encoded[col])
        
        return X_encoded
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def save_preprocessors(self, save_dir):
        """Save encoders and scaler"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.label_encoders, save_dir / 'label_encoders.pkl')
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        joblib.dump(self.le_target, save_dir / 'le_target.pkl')
    
    def load_preprocessors(self, load_dir):
        """Load saved preprocessors"""
        load_dir = Path(load_dir)
        
        self.label_encoders = joblib.load(load_dir / 'label_encoders.pkl')
        self.scaler = joblib.load(load_dir / 'scaler.pkl')
        self.le_target = joblib.load(load_dir / 'le_target.pkl')