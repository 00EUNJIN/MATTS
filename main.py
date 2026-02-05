import os
import json
import time
import torch
import torch.optim as optim
import numpy as np

from config import get_args
from model import MATTS
from dataloader import load_and_preprocess_data, prepare_dataloaders
from train import train_epoch, validate
from utils import (
    EarlyStopping, 
    setup_logging, 
    count_parameters, 
    get_model_size,
    compute_abnormal_class_weights,
    evaluate_model
)


def train(args):
    """Training pipeline"""
    
    # Setup
    save_dir = os.path.join(args.save_dir, f'th{args.anomaly_threshold}')
    logger = setup_logging(save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info('Loading and preprocessing data...')
    data = load_and_preprocess_data(args.data_path)
    data_loaders = prepare_dataloaders(
        data, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size
    )
    
    # Initialize model
    logger.info('Initializing MATTS model...')
    model = MATTS(
        input_dim=data_loaders['input_size'],
        hidden_dim=args.hidden_size,
        seq_length=args.seq_length,
        anomaly_threshold=args.anomaly_threshold
    ).to(device)
    
    # Model info
    param_info = count_parameters(model)
    size_info = get_model_size(model)
    
    logger.info("=" * 60)
    logger.info("Model Information")
    logger.info("=" * 60)
    logger.info(f"Total Parameters: {param_info['total_parameters']:,}")
    logger.info(f"Trainable Parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"Model Size: {size_info['total_size_mb']:.2f} MB")
    logger.info("=" * 60)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_path=os.path.join(save_dir, 'best_model.pt')
    )
    
    # Class weights
    abnormal_class_weights = compute_abnormal_class_weights(
        data_loaders['train_loader'], device
    )
    
    # Loss config
    loss_config = {
        'recon_weight': args.recon_weight,
        'anomaly_weight': args.anomaly_weight,
        'class_weight': args.class_weight,
    }
    
    # Training loop
    logger.info('Starting training...')
    train_losses, val_losses = [], []
    epoch_times = []
    
    total_train_start = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_result = train_epoch(
            model, data_loaders['train_loader'], optimizer, 
            device, loss_config, abnormal_class_weights
        )
        
        # Validate
        val_result = validate(
            model, data_loaders['val_loader'], 
            device, loss_config, abnormal_class_weights
        )
        
        epoch_time = train_result['epoch_time']
        epoch_times.append(epoch_time)
        train_losses.append(train_result['total_loss'])
        val_losses.append(val_result['total_loss'])
        
        # Logging
        logger.info(f'\nEpoch {epoch+1}/{args.epochs} (Time: {epoch_time:.2f}s):')
        logger.info(
            f'Train - Total: {train_result["total_loss"]:.4f}, '
            f'Recon: {train_result["recon_loss"]:.4f}, '
            f'Anomaly: {train_result["anomaly_loss"]:.4f}, '
            f'Class: {train_result["class_loss"]:.4f}'
        )
        logger.info(
            f'Val - Total: {val_result["total_loss"]:.4f}, '
            f'Recon: {val_result["recon_loss"]:.4f}, '
            f'Anomaly: {val_result["anomaly_loss"]:.4f}, '
            f'Class: {val_result["class_loss"]:.4f}, '
            f'F1_macro: {val_result["f1_macro"]:.4f}'
        )
        
        # ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining = args.epochs - epoch - 1
        logger.info(f'ETA: {(avg_epoch_time * remaining) / 60:.1f} minutes')
        
        # Early stopping
        early_stopping(val_result['f1_macro'], model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    total_train_time = time.time() - total_train_start
    
    # Training summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Total Time: {total_train_time:.2f}s ({total_train_time/60:.2f} min)")
    logger.info(f"Epochs Trained: {len(epoch_times)}")
    logger.info(f"Avg Epoch Time: {np.mean(epoch_times):.2f}s")
    logger.info("=" * 60)
    
    # Load best model and evaluate
    logger.info('Loading best model for evaluation...')
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    
    logger.info('Evaluating on test set...')
    test_results = evaluate_model(model, data_loaders['test_loader'], device)
    
    if test_results:
        # Add training info
        test_results['training_info'] = {
            'total_time_seconds': total_train_time,
            'total_epochs': len(epoch_times),
            'parameters': param_info,
            'model_size': size_info
        }
        
        # Save results
        results_path = os.path.join(save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Print results
        logger.info('\nTest Results:')
        logger.info(f"Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"F1 Macro: {test_results['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {test_results['f1_weighted']:.4f}")
        
        if 'anomaly_detection' in test_results:
            logger.info("\nAnomaly Detection:")
            for k, v in test_results['anomaly_detection'].items():
                logger.info(f"  {k}: {v:.4f}")
    
    return test_results


def inference(args):
    """Inference pipeline"""
    
    # Setup
    save_dir = os.path.join(args.save_dir, f'th{args.anomaly_threshold}')
    logger = setup_logging(save_dir, 'inference.log')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    
    # Load data
    logger.info('Loading and preprocessing data...')
    data = load_and_preprocess_data(args.data_path)
    data_loaders = prepare_dataloaders(
        data, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size
    )
    
    # Load model
    model_path = os.path.join(save_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    logger.info('Loading model...')
    model = MATTS(
        input_dim=data_loaders['input_size'],
        hidden_dim=args.hidden_size,
        seq_length=args.seq_length,
        anomaly_threshold=args.anomaly_threshold
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Model info
    param_info = count_parameters(model)
    size_info = get_model_size(model)
    
    logger.info(f"Parameters: {param_info['total_parameters']:,}")
    logger.info(f"Model Size: {size_info['total_size_mb']:.2f} MB")
    
    # Warm-up
    test_loader = data_loaders['test_loader']
    with torch.no_grad():
        for data_batch, _ in test_loader:
            _ = model(data_batch.to(device))
            break
    
    # Measure inference time
    batch_times = []
    total_samples = 0
    
    logger.info("Measuring inference time...")
    with torch.no_grad():
        for data_batch, _ in test_loader:
            data_batch = data_batch.to(device)
            
            start = time.time()
            _ = model(data_batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            batch_times.append(end - start)
            total_samples += data_batch.size(0)
    
    total_time = sum(batch_times)
    avg_sample_time = total_time / total_samples
    
    logger.info("=" * 60)
    logger.info("Inference Time")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Total Time: {total_time:.4f}s")
    logger.info(f"Avg per Sample: {avg_sample_time*1000:.4f}ms")
    logger.info(f"Throughput: {1/avg_sample_time:.2f} samples/sec")
    logger.info("=" * 60)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    if results:
        results['inference_time'] = {
            'total_samples': total_samples,
            'total_time_seconds': total_time,
            'avg_sample_time_ms': avg_sample_time * 1000,
            'throughput': 1 / avg_sample_time
        }
        
        # Save
        output_path = os.path.join(save_dir, 'inference_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print
        print("\n=== Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Macro: {results['f1_macro']:.4f}")
        print(f"Throughput: {results['inference_time']['throughput']:.2f} samples/sec")
    
    return results


def main():
    args = get_args()
    
    print("\n" + "=" * 50)
    print("TwoStageLSTMAE - Multi-class Anomaly Detection")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Threshold: {args.anomaly_threshold}")
    print("=" * 50 + "\n")
    
    if args.mode == 'train':
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
