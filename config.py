import argparse


def get_args():
    parser = argparse.ArgumentParser(description='TwoStageLSTMAE for Multi-class Anomaly Detection')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'inference'],
                        help='Mode: train or inference')
    
    # Data
    parser.add_argument('--data_path', type=str, 
                        default='data/tcm5_dataset_4.csv',
                        help='Path to the dataset file')
    
    # Model
    parser.add_argument('--seq_length', type=int, default=5,
                        help='Length of each sequence')
    parser.add_argument('--hidden_size', type=int, default=16,
                        help='Hidden layer size')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5,
                        help='Threshold for anomaly detection (0~1)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--anomaly_weight', type=float, default=10.0,
                        help='Anomaly detection loss weight')
    parser.add_argument('--class_weight', type=float, default=10.0,
                        help='Classification loss weight')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save the model')
    
    args = parser.parse_args()
    
    return args


# Default configuration
DEFAULT_CONFIG = {
    'seq_length': 5,
    'hidden_size': 16,
    'anomaly_threshold': 0.5,
    'batch_size': 4,
    'epochs': 1000,
    'lr': 0.0005,
    'patience': 10,
    'min_delta': 0.001,
    'recon_weight': 1.0,
    'anomaly_weight': 10.0,
    'class_weight': 10.0,
}