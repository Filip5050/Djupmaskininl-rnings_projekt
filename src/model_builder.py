"""
PyTorch Neural Network model builder for fraud detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from src.config import Config


class FocalLoss(nn.Module):
    """Focal Loss for handling imbalanced classification
    
    Focal Loss focuses training on hard examples and down-weights easy examples.
    This is especially effective for fraud detection where frauds are rare.
    
    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        alpha: Weighting factor for positive class (fraud)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Probability of correct class
        pt = torch.exp(-bce_loss)
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class FraudDetector(nn.Module):
    """Deep Neural Network for fraud detection (binary classification)"""
    
    def __init__(self, input_dim: int):
        super(FraudDetector, self).__init__()
        
        # Layer 1: 128 neurons
        self.fc1 = nn.Linear(input_dim, Config.LAYER_SIZES[0])
        self.bn1 = nn.BatchNorm1d(Config.LAYER_SIZES[0])
        self.dropout1 = nn.Dropout(Config.DROPOUT_RATES[0])
        
        # Layer 2: 64 neurons
        self.fc2 = nn.Linear(Config.LAYER_SIZES[0], Config.LAYER_SIZES[1])
        self.bn2 = nn.BatchNorm1d(Config.LAYER_SIZES[1])
        self.dropout2 = nn.Dropout(Config.DROPOUT_RATES[1])
        
        # Layer 3: 32 neurons
        self.fc3 = nn.Linear(Config.LAYER_SIZES[1], Config.LAYER_SIZES[2])
        self.bn3 = nn.BatchNorm1d(Config.LAYER_SIZES[2])  # Added batch norm
        self.dropout3 = nn.Dropout(Config.DROPOUT_RATES[2])
        
        # Layer 4: 16 neurons
        self.fc4 = nn.Linear(Config.LAYER_SIZES[2], Config.LAYER_SIZES[3])
        self.dropout4 = nn.Dropout(Config.DROPOUT_RATES[3])
        
        # Output layer: binary classification
        self.fc5 = nn.Linear(Config.LAYER_SIZES[3], 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)  # Batch norm added
        x = self.relu(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout4(x)
        
        # Output
        x = self.fc5(x)
        x = self.sigmoid(x)
        
        return x


def build_fraud_classifier(input_dim: int) -> FraudDetector:
    """Build a deep neural network for fraud detection"""
    print("\n🏗️  Building PyTorch Fraud Detection Model...")
    
    model = FraudDetector(input_dim)
    
    # Print model architecture
    print("✅ Model built successfully")
    print(f"   Architecture: {' → '.join(map(str, Config.LAYER_SIZES))} → 1")
    print(f"\n📊 Model Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def save_model(model: FraudDetector, path: Path):
    """Save PyTorch model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'layer_sizes': Config.LAYER_SIZES,
            'dropout_rates': Config.DROPOUT_RATES
        }
    }, path)
    print(f"✅ Model saved to: {path}")


def load_model(path: Path, input_dim: int) -> FraudDetector:
    """Load PyTorch model"""
    print(f"📂 Loading model from: {path}")
    
    model = FraudDetector(input_dim)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded")
    return model
