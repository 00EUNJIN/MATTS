import argparse

def get_args():
    parser = argparse.ArgumentParser(description='TwoStageLSTMAE Training')
    
    parser.add_argument('--data_path', type=str, default='./data/dataset.csv', help='Path to dataset')
    parser.add_argument('--seq_length', type=int, default=5, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--anomaly_threshold', type=float, default=0.5, help='Anomaly threshold')
    
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Min delta')
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    return args