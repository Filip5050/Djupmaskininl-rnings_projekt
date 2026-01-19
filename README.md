# Credit Card Fraud Detection System

Deep learning-based system for credit card fraud detection using PyTorch.

## Overview

This project uses a 4-layer neural network with Focal Loss to handle extremely imbalanced data (0.173% fraud rate). The system achieves aroudn 92% recall and AUC around 0.97 on test data.


## Dataset

**IMPORTANT**: The dataset is not included in this repository due to its size (143 MB).

### Download the dataset:

1. Go to Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place the file here: `data/fraud_detection/creditcard.csv`

```bash
# Create directory if it doesn't exist
mkdir -p data/fraud_detection

# Move the downloaded file
mv ~/Downloads/creditcard.csv data/fraud_detection/
```

## Installation

### 1. Clone the project
```bash
git clone https://github.com/Filip5050/Djupmaskininl-rnings_projekt.git
cd Djupmaskininl-rnings_projekt
```



### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset (see above)

## Usage

### Train the model
```bash
python main.py train
```

This will:
- Load and preprocess data
- Train the model with Focal Loss
- Save the best model in `models/saved_models/`
- Display training history and metrics

### Evaluate the model
```bash
python main.py evaluate
```

Generates:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Probability distribution
- Saves visualization: `fraud_evaluation.png`

### Launch web app
```bash
python main.py webapp
```

Opens Streamlit dashboard at http://localhost:8501 with:
- **Overview**: Training metrics, confusion matrix, business impact
- **Fraud Detection**: Test individual transactions
- **About**: Project information

## Architecture

### Model
- **Layers**: 128 → 64 → 32 → 16 → 1
- **Activation**: ReLU
- **Dropout**: [0.4, 0.35, 0.3, 0.25]
- **BatchNormalization**: On layers 1-3
- **Parameters**: 15,297

### Training
- **Loss Function**: Focal Loss (alpha=0.25, gamma=2.0)
- **Optimizer**: Adam (lr=0.0005)
- **Batch Size**: 256
- **Class Weights**: fraud=900, normal=1 (WeightedRandomSampler)
- **Early Stopping**: Patience=8
- **LR Scheduler**: ReduceLROnPlateau (patience=3)

### Preprocessing
- **Scaler**: RobustScaler (Time + Amount only)
- **Features**: V1-V28 already PCA-transformed
- **SMOTE**: Disabled (creates unrealistic patterns)

### Classification
- **Threshold**: 0.40 (optimized for high recall)

## Project Structure

```
.
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── data_loader.py      # Data loading and splitting
│   ├── preprocessor.py     # Feature scaling
│   ├── model_builder.py    # PyTorch model + Focal Loss
│   ├── train.py            # Training pipeline
│   └── evaluate.py         # Evaluation and visualization
├── web_app/
│   └── app.py              # Streamlit dashboard
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md               # This file
```


## Requirements

See `requirements.txt` for complete list. Main dependencies:
- Python 3.8+
- PyTorch 2.0+
- scikit-learn
- pandas
- numpy
- matplotlib
- streamlit
- joblib

## License

This is a school project for deep learning course.

## Author

Filip5050 - [GitHub](https://github.com/Filip5050)

## Acknowledgments

- Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
