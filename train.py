import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_epoch(model, train_loader, optimizer, device, config, abnormal_class_weights=None):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        config: Configuration dictionary with loss weights
        abnormal_class_weights: Class weights for imbalanced data
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    total_loss = 0
    
    recon_weight = config.get('recon_weight', 1.0)
    anomaly_weight = config.get('anomaly_weight', 10.0)
    class_weight = config.get('class_weight', 10.0)
    
    epoch_start_time = time.time()
    
    loss_tracking = {
        'recon_loss': 0,
        'anomaly_loss': 0,
        'class_loss': 0
    }
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            decoded, anomaly_score, class_score = model(data)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(decoded, data) * recon_weight
            loss = recon_loss
            
            # Anomaly detection loss
            anomaly_target = (target > 0).float()
            anomaly_loss = nn.BCELoss()(anomaly_score.flatten(), anomaly_target.flatten()) * anomaly_weight
            loss += anomaly_loss
            
            # Classification loss (only for anomalies)
            anomaly_mask = target > 0
            if anomaly_mask.any():
                valid_labels = target[anomaly_mask] - 1
                masked_class_score = class_score[anomaly_mask]
                
                if abnormal_class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=abnormal_class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
                
                class_loss_val = criterion(masked_class_score, valid_labels) * class_weight
                loss += class_loss_val
                loss_tracking['class_loss'] += class_loss_val.item()
            
            loss_tracking['recon_loss'] += recon_loss.item()
            loss_tracking['anomaly_loss'] += anomaly_loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'Anomaly': f'{anomaly_loss.item():.4f}'
            })
    
    epoch_time = time.time() - epoch_start_time
    num_batches = len(train_loader)
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': loss_tracking['recon_loss'] / num_batches,
        'anomaly_loss': loss_tracking['anomaly_loss'] / num_batches,
        'class_loss': loss_tracking['class_loss'] / num_batches,
        'epoch_time': epoch_time
    }


def validate(model, val_loader, device, config, abnormal_class_weights=None):
    """
    Validate model
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        device: Device (cuda/cpu)
        config: Configuration dictionary with loss weights
        abnormal_class_weights: Class weights for imbalanced data
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    
    recon_weight = config.get('recon_weight', 1.0)
    anomaly_weight = config.get('anomaly_weight', 10.0)
    class_weight = config.get('class_weight', 10.0)
    
    loss_tracking = {
        'recon_loss': 0,
        'anomaly_loss': 0,
        'class_loss': 0
    }
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            decoded, anomaly_score, class_score = model(data)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(decoded, data) * recon_weight
            loss = recon_loss
            
            # Anomaly detection loss
            anomaly_target = (target > 0).float()
            anomaly_loss = nn.BCELoss()(anomaly_score.squeeze(), anomaly_target) * anomaly_weight
            loss += anomaly_loss
            
            # Classification loss
            anomaly_mask = target > 0
            if anomaly_mask.any():
                valid_labels = target[anomaly_mask] - 1
                masked_class_score = class_score[anomaly_mask]
                
                if abnormal_class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=abnormal_class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
                
                class_loss_val = criterion(masked_class_score, valid_labels) * class_weight
                loss += class_loss_val
                loss_tracking['class_loss'] += class_loss_val.item()
            
            loss_tracking['recon_loss'] += recon_loss.item()
            loss_tracking['anomaly_loss'] += anomaly_loss.item()
            total_loss += loss.item()
            
            # Predictions for F1 calculation
            anomaly_pred = (anomaly_score.squeeze(-1) > model.anomaly_threshold).long()
            if anomaly_pred.dim() == 0:
                anomaly_pred = anomaly_pred.view(1)
            
            _, class_pred = torch.max(class_score, 1)
            
            final_pred = torch.zeros(batch_size, device=device, dtype=torch.long)
            anomaly_indices = (anomaly_pred == 1).nonzero(as_tuple=True)[0]
            if len(anomaly_indices) > 0:
                final_pred[anomaly_indices] = class_pred[anomaly_indices] + 1
            
            all_preds.extend(final_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    num_batches = len(val_loader)
    f1_macro = f1_score(np.array(all_targets), np.array(all_preds), average='macro')
    
    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': loss_tracking['recon_loss'] / num_batches,
        'anomaly_loss': loss_tracking['anomaly_loss'] / num_batches,
        'class_loss': loss_tracking['class_loss'] / num_batches,
        'f1_macro': f1_macro
    }