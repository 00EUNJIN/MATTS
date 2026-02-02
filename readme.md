# MATTS: A Hierarchical Framework for Multi-class Time-series Anomaly Detection in Industrial Manufacturing

Official implementation of **MATTS**, a two-stage LSTM Autoencoder framework for
multi-class anomaly detection in industrial multivariate time-series data.

---

## Model Overview

MATTS consists of two decoupled stages:

- **Stage 1 (Detection):**  
  An LSTM Autoencoder trained to minimize reconstruction error, optimized for
  rapid and sensitive anomaly detection.

- **Stage 2 (Diagnosis):**  
  A multi-class classifier trained only on anomalous samples to distinguish
  fault types without interference from dominant normal patterns.

This hierarchical design explicitly decouples detection and diagnosis objectives,
mitigating optimization conflicts under severe class imbalance.

---

## Architecture Details (Common)

### LSTM Autoencoder
- Window length (sequence length): 5
- Sliding window stride: 1
- Encoder–decoder: single-layer LSTM
- Reconstruction loss: Mean Squared Error (MSE)

### Diagnostic Classifier
- Input: latent representation from LSTM encoder
- Architecture: lightweight MLP
- Loss: weighted cross-entropy

---

## Dataset-specific Best Hyperparameters

The optimal hyperparameters were independently selected for each dataset
based on validation Macro F1-score.

| Hyperparameter | Dataset 4 | Dataset 5 | Dataset 6 |
|---------------|-----------|-----------|-----------|
| Hidden size | 16 | 32 | 16 |
| Sequence length | 5 | 5 | 5 |
| Learning rate | 0.0005 | 0.0001 | 0.0005 |
| Batch size | 4 | 4 | 4 |
| Reconstruction loss weight | 1 | 1 | 1 |
| Anomaly loss weight | 5 | 5 | 10 |
| Class loss weight | 10 | 15 | 15 |
| Learned threshold (τ) | 0.5 | 0.5 | 0.5 |

---

## Training Configuration

- Optimizer: Adam
- Maximum epochs: 1000
- Early stopping: enabled (patience = 10)
- Loss weighting: enabled to address severe class imbalance

No explicit dropout or L2 regularization is applied.

---

## Computational Information

| Hidden Size | Trainable Parameters | Model Size |
|------------:|---------------------:|-----------:|
| 16 | 4,773 | 0.018 MB |
| 32 | 12,101 | 0.046 MB |

Training time and convergence behavior vary across datasets and are reported
in the experimental section of the paper.

---

## Usage

### Training
```bash
python train.py \
  --data_path ./data/dataset.csv \
  --seq_length 5 \
  --batch_size 4 \
  --hidden_size 16 \
  --epochs 1000 \
  --save_dir ./checkpoints
````

### Inference

```bash
python inference.py \
  --data_path ./data/dataset.csv \
  --save_dir ./checkpoints
```

---

## Output Files

* `best_model.pt`: trained MATTS weights
* `training.log`: training logs
* `results.json`: evaluation metrics
* `inference_results.json`: inference outputs

```

