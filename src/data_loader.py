"""
Data loading utilities for fraud detection
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config


def load_fraud_data():
    print(f"Loading fraud data from: {Config.DATA_FILE}")
    df = pd.read_csv(Config.DATA_FILE)
    
    print(f"Loaded {len(df):,} transactions")
    
    # Class distribution
    fraud_count = df[Config.TARGET_COL].sum()
    normal_count = len(df) - fraud_count
    fraud_pct = fraud_count / len(df) * 100
    normal_pct = normal_count / len(df) * 100
    
    print(f"   Normal transactions: {normal_count:,} ({normal_pct:.2f}%)")
    print(f"   Fraudulent transactions: {fraud_count:,} ({fraud_pct:.2f}%)")
    
    return df


def split_train_test(df):
    print("\nSplitting data...")
    train_df, test_df = train_test_split(
        df, 
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=df[Config.TARGET_COL]
    )
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    return train_df, test_df


def split_features_target(df):
    X = df[Config.FEATURE_COLS]
    y = df[Config.TARGET_COL]
    
    print(f"Features shape: {X.shape} | Target shape: {y.shape}")
    
    return X, y
