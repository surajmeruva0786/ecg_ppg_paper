"""
Deep Learning models for heart attack prediction using ECG and PPG signals.
"""

import os
# Force CPU mode to avoid CUDA DLL loading issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from typing import Dict, Tuple, Any
import logging
import time

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3, num_classes=2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class CNN1DClassifier(nn.Module):
    """1D CNN for time-series classification."""
    
    def __init__(self, input_channels=1, seq_length=100, filters=[64, 128, 256], 
                 kernel_size=3, pool_size=2, dropout=0.3, num_classes=2):
        super(CNN1DClassifier, self).__init__()
        
        layers = []
        in_channels = input_channels
        current_length = seq_length
        
        for num_filters in filters:
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size))
            layers.append(nn.Dropout(dropout))
            in_channels = num_filters
            current_length = current_length // pool_size
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flatten_size = filters[-1] * current_length
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_length) -> (batch, 1, seq_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class LSTMClassifier(nn.Module):
    """LSTM for time-series classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3, num_classes=2):
        super(LSTMClassifier, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.lstm_layers.append(nn.LSTM(prev_dim, hidden_dim, batch_first=True))
            prev_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_length) -> (batch, seq_length, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = self.dropout(x)
        
        # Take last output
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GRUClassifier(nn.Module):
    """GRU for time-series classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3, num_classes=2):
        super(GRUClassifier, self).__init__()
        
        self.gru_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.gru_layers.append(nn.GRU(prev_dim, hidden_dim, batch_first=True))
            prev_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_length) -> (batch, seq_length, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        
        for gru in self.gru_layers:
            x, _ = gru(x)
            x = self.dropout(x)
        
        # Take last output
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def train_deep_learning_model(
    model, X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device=None
) -> Tuple[nn.Module, Dict]:
    """
    Train a deep learning model.
    """
    # Determine device with better error handling
    if device is None:
        try:
            if torch.cuda.is_available():
                device = 'cuda'
                # Test CUDA availability
                torch.zeros(1).cuda()
        except Exception as e:
            logger.warning(f"CUDA not available or failed: {str(e)}. Using CPU instead.")
            device = 'cpu'
    
    logger.info(f"Training {model.__class__.__name__} on {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, y_pred = torch.max(val_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        y_val_np = y_val_tensor.cpu().numpy()
        
        # Get probabilities for ROC-AUC
        y_proba = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
    
    metrics = {
        'model_name': model.__class__.__name__,
        'train_time': train_time,
        'accuracy': accuracy_score(y_val_np, y_pred),
        'precision': precision_score(y_val_np, y_pred, average='weighted'),
        'recall': recall_score(y_val_np, y_pred, average='weighted'),
        'f1': f1_score(y_val_np, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_val_np, y_proba) if len(np.unique(y_val_np)) == 2 else 0,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    logger.info(f"{model.__class__.__name__} - F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics


def train_all_dl_models(
    X_train, y_train, X_val, y_val,
    models_to_train=None,
    save_dir='results/models',
    epochs=100,
    batch_size=32,
    learning_rate=0.001
) -> Dict:
    """
    Train all deep learning models.
    """
    if models_to_train is None:
        models_to_train = ['mlp', 'cnn1d', 'lstm', 'gru']
    
    results = {}
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Determine device with error handling
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            torch.zeros(1).cuda()  # Test CUDA
            logger.info(f"Using device: {device}")
        else:
            device = 'cpu'
            logger.info(f"CUDA not available. Using device: {device}")
    except Exception as e:
        device = 'cpu'
        logger.warning(f"CUDA failed: {str(e)}. Using device: {device}")
    
    # Define models
    models = {}
    if 'mlp' in models_to_train:
        models['mlp'] = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    
    if 'cnn1d' in models_to_train:
        models['cnn1d'] = CNN1DClassifier(seq_length=input_dim, num_classes=num_classes)
    
    if 'lstm' in models_to_train:
        models['lstm'] = LSTMClassifier(input_dim=1, num_classes=num_classes)
    
    if 'gru' in models_to_train:
        models['gru'] = GRUClassifier(input_dim=1, num_classes=num_classes)
    
    # Train each model
    for model_name, model in models.items():
        try:
            trained_model, metrics = train_deep_learning_model(
                model, X_train, y_train, X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device
            )
            
            results[model_name] = {
                'model': trained_model,
                'metrics': metrics
            }
            
            # Save model
            model_file = save_path / f'{model_name}.pth'
            torch.save(trained_model.state_dict(), model_file)
            logger.info(f"Saved {model_name} to {model_file}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Testing Deep Learning Models")
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test MLP
    mlp = MLPClassifier(input_dim=100, num_classes=2)
    trained_mlp, metrics = train_deep_learning_model(
        mlp, X_train, y_train, X_val, y_val,
        epochs=10,
        batch_size=32
    )
    
    print("\nMLP Results:")
    for k, v in metrics.items():
        if k not in ['train_losses', 'val_losses']:
            print(f"  {k}: {v}")
