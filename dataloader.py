import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class AnomalyDataset(Dataset):
    """Custom Dataset for Anomaly Detection"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_data(file_path):
    """
    Load and preprocess TCM dataset
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    data = pd.read_csv(file_path)
    
    # Convert boolean to int
    for col in data.columns:
        if col.startswith('Anomaly'):
            data[col] = data[col].replace({False: 0, True: 1})
    
    # Aggregate anomaly columns
    electric_columns = [col for col in data.columns if col.startswith('Anomaly_Electric_')]
    bearing_columns = [col for col in data.columns if col.startswith('Anomaly_Bearing_')]
    workroll_columns = [col for col in data.columns if col.startswith('Anomaly_WorkRoll_')]
    
    data['Anomaly_Electric'] = data[electric_columns].any(axis=1).astype(int)
    data['Anomaly_Bearing'] = data[bearing_columns].any(axis=1).astype(int)
    data['Anomaly_WorkRoll'] = data[workroll_columns].any(axis=1).astype(int)
    
    # Drop original columns
    columns_to_drop = electric_columns + bearing_columns + workroll_columns
    data = data.drop(columns=columns_to_drop)
    
    return data


def create_sequences(data, seq_length):
    """Create sequences from data"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)


def prepare_dataloaders(data, seq_length=10, batch_size=16, train_ratio=0.7, val_ratio=0.1):
    """
    Prepare train, validation, and test dataloaders
    
    Args:
        data: Preprocessed DataFrame
        seq_length: Sequence length
        batch_size: Batch size
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        
    Returns:
        Dictionary containing dataloaders and metadata
    """
    # Separate features and labels
    X = data.drop(['Anomaly_Reduction', 'Anomaly_Electric', 'Anomaly_Bearing', 'Anomaly_WorkRoll'], axis=1)
    
    # Create labels (0: Normal, 1: Reduction, 2: Electric, 3: Bearing, 4: WorkRoll)
    y = np.zeros(len(data))
    y[data['Anomaly_Reduction'] == 1] = 1
    y[data['Anomaly_Electric'] == 1] = 2
    y[data['Anomaly_Bearing'] == 1] = 3
    y[data['Anomaly_WorkRoll'] == 1] = 4
    
    # Create sequences
    X_sequences = create_sequences(X.values, seq_length)
    y_sequences = y[seq_length-1:]
    
    print(f'Sequence Length: {seq_length}')
    print(f'Batch Size: {batch_size}')
    print(f'Total Sequences: {len(X_sequences)}')
    
    # Split data
    total_length = len(X_sequences)
    train_size = int(train_ratio * total_length)
    val_size = int(val_ratio * total_length)
    
    # Normalize
    scaler = StandardScaler()
    X_sequences_reshaped = X_sequences.reshape(len(X_sequences), -1)
    X_sequences_scaled = scaler.fit_transform(X_sequences_reshaped)
    X_sequences = X_sequences_scaled.reshape(X_sequences.shape)
    
    # Split
    X_train = X_sequences[:train_size]
    X_val = X_sequences[train_size:train_size+val_size]
    X_test = X_sequences[train_size+val_size:]
    
    y_train = y_sequences[:train_size]
    y_val = y_sequences[train_size:train_size+val_size]
    y_test = y_sequences[train_size+val_size:]
    
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    
    # Create datasets
    train_dataset = AnomalyDataset(X_train, y_train)
    val_dataset = AnomalyDataset(X_val, y_val)
    test_dataset = AnomalyDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'input_size': X.shape[1],
        'scaler': scaler
    }