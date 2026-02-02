import os
import logging
import copy
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        checkpoint_path: Path to save best model
    """
    
    def __init__(self, patience=10, min_delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_path = checkpoint_path
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_model = copy.deepcopy(model)

    def load_best_model(self, model):
        if self.checkpoint_path:
            model.load_state_dict(torch.load(self.checkpoint_path))
        return model


def setup_logging(save_dir, log_name='training.log'):
    """Setup logging configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, log_name)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }


def compute_abnormal_class_weights(train_loader, device):
    """Compute class weights for imbalanced data"""
    all_labels = []
    for _, target in train_loader:
        abnormal_mask = target > 0
        if abnormal_mask.any():
            abnormal_labels = target[abnormal_mask] - 1
            all_labels.extend(abnormal_labels.numpy())
    
    if len(all_labels) > 0:
        all_labels = np.array(all_labels)
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        weights = total_samples / (len(class_counts) * class_counts)
        weights = weights / np.sum(weights)
        return torch.FloatTensor(weights).to(device)
    
    return None


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        data_loader: Test dataloader
        device: Device (cuda/cpu)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            decoded, anomaly_score, class_score = model(data)
            
            # Handle anomaly score dimensions
            if anomaly_score.dim() == 0:
                anomaly_score = anomaly_score.view(1, 1)
            elif anomaly_score.dim() == 1:
                if anomaly_score.size(0) == 1:
                    anomaly_score = anomaly_score.view(1, 1)
                else:
                    anomaly_score = anomaly_score.view(-1, 1)
            
            # Anomaly prediction
            anomaly_pred = (anomaly_score.squeeze(-1) > model.anomaly_threshold).long()
            if anomaly_pred.dim() == 0:
                anomaly_pred = anomaly_pred.view(1)
            
            # Class prediction
            _, class_pred = torch.max(class_score, 1)
            
            # Final prediction
            final_pred = torch.zeros(batch_size, device=device, dtype=torch.long)
            anomaly_indices = (anomaly_pred == 1).nonzero(as_tuple=True)[0]
            if len(anomaly_indices) > 0:
                final_pred[anomaly_indices] = class_pred[anomaly_indices] + 1
            
            all_preds.extend(final_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Collect anomaly scores
            anomaly_score_np = anomaly_score.cpu().numpy()
            if np.isscalar(anomaly_score_np) or anomaly_score_np.ndim == 0:
                all_anomaly_scores.append(float(anomaly_score_np))
            elif anomaly_score_np.ndim == 1:
                all_anomaly_scores.extend(anomaly_score_np)
            else:
                all_anomaly_scores.extend(anomaly_score_np.flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_anomaly_scores = np.array(all_anomaly_scores)
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1_macro': f1_score(all_targets, all_preds, average='macro'),
        'f1_micro': f1_score(all_targets, all_preds, average='micro'),
        'f1_weighted': f1_score(all_targets, all_preds, average='weighted'),
        'per_class_metrics': {}
    }
    
    # Anomaly detection metrics
    anomaly_true = (all_targets > 0).astype(int)
    anomaly_pred = (all_preds > 0).astype(int)
    
    if len(np.unique(anomaly_true)) > 1:
        results['anomaly_detection'] = {
            'accuracy': accuracy_score(anomaly_true, anomaly_pred),
            'precision': precision_score(anomaly_true, anomaly_pred, zero_division=0),
            'recall': recall_score(anomaly_true, anomaly_pred),
            'f1': f1_score(anomaly_true, anomaly_pred)
        }
        
        if len(all_anomaly_scores) > 0:
            try:
                results['anomaly_detection']['auc'] = roc_auc_score(anomaly_true, all_anomaly_scores)
            except ValueError:
                results['anomaly_detection']['auc'] = 0
    
    # Per-class metrics
    for i in np.unique(np.concatenate([all_targets, all_preds])):
        binary_targets = (all_targets == i).astype(int)
        binary_preds = (all_preds == i).astype(int)
        
        results['per_class_metrics'][f'Class_{int(i)}'] = {
            'precision': precision_score(binary_targets, binary_preds, zero_division=0),
            'recall': recall_score(binary_targets, binary_preds, zero_division=0),
            'f1': f1_score(binary_targets, binary_preds, zero_division=0)
        }
    
    return results