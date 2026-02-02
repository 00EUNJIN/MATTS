import os
import json
import torch
import logging
import time
from options import get_args
from dataloader import load_and_preprocess_data, prepare_dataloaders
from model import TwoStageLSTMAE
from utils import evaluate_model, count_parameters, get_model_size

def setup_logging(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    args = get_args()
    logger = setup_logging(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f'Device: {device}')
    
    data = load_and_preprocess_data(args.data_path)
    data_loaders = prepare_dataloaders(
        data,
        seq_length=args.seq_length,
        batch_size=args.batch_size
    )
    
    model_path = os.path.join(args.save_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    model = TwoStageLSTMAE(
        input_dim=data_loaders['input_size'],
        seq_length=args.seq_length,
        hidden_dim=args.hidden_size,
        anomaly_threshold=args.anomaly_threshold
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    param_info = count_parameters(model)
    size_info = get_model_size(model)
    
    logger.info("="*60)
    logger.info(f"Total Parameters: {param_info['total_parameters']:,}")
    logger.info(f"Model Size: {size_info['total_size_mb']:.2f} MB")
    logger.info("="*60)
    
    test_loader = data_loaders['test_loader']
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            _ = model(data)
            break
    
    batch_times = []
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            batch_start = time.time()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_end = time.time()
            
            batch_times.append(batch_end - batch_start)
            total_samples += data.size(0)
    
    total_time = sum(batch_times)
    avg_sample_time = total_time / total_samples
    
    logger.info("="*60)
    logger.info("Inference Time Statistics")
    logger.info("="*60)
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Total Time: {total_time:.4f}s")
    logger.info(f"Avg Time per Sample: {avg_sample_time*1000:.4f}ms")
    logger.info(f"Throughput: {1/avg_sample_time:.2f} samples/sec")
    logger.info("="*60)
    
    results = evaluate_model(model, test_loader, device)
    
    results['model_info'] = {
        'parameters': param_info,
        'model_size_mb': size_info['total_size_mb'],
        'inference': {
            'avg_sample_time_ms': avg_sample_time * 1000,
            'throughput_samples_per_sec': 1 / avg_sample_time
        }
    }
    
    output_path = os.path.join(args.save_dir, 'inference_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Macro: {results['f1_macro']:.4f}")

if __name__ == "__main__":
    main()