import torch
import torch.nn as nn


class TwoStageLSTMAutoencoder(nn.Module):
    """
    Two-Stage LSTM Autoencoder for Multi-class Anomaly Detection
    
    Stage 1: Anomaly Detection (Binary Classification)
    Stage 2: Anomaly Type Classification (Multi-class)
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for LSTM layers
        seq_length: Length of input sequences
        anomaly_threshold: Threshold for anomaly detection (default: 0.5)
    """
    
    def __init__(self, input_dim, hidden_dim, seq_length, anomaly_threshold=0.5):
        super(TwoStageLSTMAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.anomaly_threshold = anomaly_threshold
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            batch_first=True
        )
        
        # Anomaly Detector (Stage 1)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Classifier (Stage 2)
        self.classifier = nn.Linear(hidden_dim, 4)
    
    def forward(self, x):
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Decode
        hidden_repeated = hidden.permute(1, 0, 2)
        hidden_repeated = hidden_repeated.repeat(1, self.seq_length, 1)
        decoded, _ = self.decoder(hidden_repeated)
        
        # Extract features from last timestep
        features = encoded[:, -1, :]
        
        # Stage 1: Anomaly Detection
        anomaly_score = self.anomaly_detector(features)
        
        # Stage 2: Classification
        class_score = self.classifier(features)
        
        return decoded, anomaly_score, class_score