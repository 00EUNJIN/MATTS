import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MATTS: A Hierarchical Framework for Multi-class Time-series Anomaly Detection in Industrial Manufacturing')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='Mode: train or inference')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data/tcm5_dataset_4.csv', help='Path to the dataset file')
    
    # Model architecture
    parser.add_argument('--seq_length', type=int, default=5, help='Length of each sequence')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--latent_dim', type=int, default=8, help='Latent dimension for autoencoder')
    
    # Anomaly detection
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='Threshold for anomaly detection (0~1)')
    parser.add_argument('--threshold_method', type=str, default='fixed', choices=['fixed', 'percentile', 'adaptive'], help='Method for determining anomaly threshold')
    parser.add_argument('--percentile', type=float, default=95.0, help='Percentile for threshold (when threshold_method=percentile)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'cosine', 'plateau'], help='Learning rate scheduler')
    parser.add_argument('--scheduler_step', type=int, default=50, help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for StepLR scheduler')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--anomaly_weight', type=float, default=10.0, help='Anomaly detection loss weight')
    parser.add_argument('--class_weight', type=float, default=10.0, help='Classification loss weight')
    
    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test data ratio')
    
    # Logging and saving
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the model')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logging')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval (epochs)')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Inference
    parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model for inference')
    parser.add_argument('--output_path', type=str, default='results', help='Path to save inference results')
    
    args = parser.parse_args()
    return args


# Default configuration
DEFAULT_CONFIG = {
    'seq_length': 5,
    'hidden_size': 16,
    'num_layers': 2,
    'dropout': 0.1,
    'latent_dim': 8,
    'anomaly_threshold': 0.5,
    'threshold_method': 'fixed',
    'percentile': 95.0,
    'batch_size': 4,
    'epochs': 1000,
    'lr': 0.0005,
    'weight_decay': 1e-5,
    'scheduler': 'none',
    'scheduler_step': 50,
    'scheduler_gamma': 0.5,
    'patience': 10,
    'min_delta': 0.001,
    'recon_weight': 1.0,
    'anomaly_weight': 10.0,
    'class_weight': 10.0,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'log_interval': 10,
    'save_interval': 50,
    'seed': 42,
}
