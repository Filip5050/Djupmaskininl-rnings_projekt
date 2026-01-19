import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_time_features(self, df):
        """Create time-based features"""
        df_new = df.copy()
        
        # Bytes per second
        df_new['bytes_per_second'] = df_new['src_bytes'] / (df_new['duration'] + 1)
        df_new['dst_bytes_per_second'] = df_new['dst_bytes'] / (df_new['duration'] + 1)
        
        # Total bytes
        df_new['total_bytes'] = df_new['src_bytes'] + df_new['dst_bytes']
        df_new['byte_ratio'] = df_new['src_bytes'] / (df_new['dst_bytes'] + 1)
        
        return df_new
    
    def create_error_features(self, df):
        """Create error-based features"""
        df_new = df.copy()
        
        # Error ratios
        df_new['error_ratio'] = df_new['serror_rate'] / (df_new['rerror_rate'] + 0.01)
        df_new['srv_error_ratio'] = df_new['srv_serror_rate'] / (df_new['srv_rerror_rate'] + 0.01)
        
        # Total error rate
        df_new['total_error_rate'] = (df_new['serror_rate'] + df_new['rerror_rate']) / 2
        
        return df_new
    
    def create_connection_features(self, df):
        """Create connection-based features"""
        df_new = df.copy()
        
        # Service patterns
        df_new['same_srv_ratio'] = df_new['same_srv_rate'] * df_new['srv_count']
        df_new['diff_srv_ratio'] = df_new['diff_srv_rate'] * df_new['count']
        
        # Host patterns
        df_new['dst_host_interaction'] = (
            df_new['dst_host_srv_count'] * df_new['dst_host_same_srv_rate']
        )
        
        return df_new
    
    def create_security_flags(self, df):
        """Create binary security flags"""
        df_new = df.copy()
        
        # Failed login flag
        df_new['has_failed_logins'] = (df_new['num_failed_logins'] > 0).astype(int)
        
        # Root access flag
        df_new['has_root_access'] = (df_new['num_root'] > 0).astype(int)
        
        # Compromised flag
        df_new['has_compromised'] = (df_new['num_compromised'] > 0).astype(int)
        
        # File creation flag
        df_new['has_file_creations'] = (df_new['num_file_creations'] > 0).astype(int)
        
        return df_new
    
    def create_all_features(self, df):
        """Apply all feature engineering"""
        df = self.create_time_features(df)
        df = self.create_error_features(df)
        df = self.create_connection_features(df)
        df = self.create_security_flags(df)
        return df