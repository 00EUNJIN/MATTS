---

# MATTS: Multi-class Anomaly Detection Through Two-Stage Strategy

MATTS is a hierarchical two-stage framework for multi-class anomaly detection in industrial time-series data, explicitly designed to address severe class imbalance and optimization conflicts between detection and diagnosis.

---

## Model Overview

MATTS consists of two decoupled stages:

* **Stage 1 (Detection):**
  An LSTM Autoencoder trained to minimize reconstruction error, optimized for rapid and sensitive anomaly detection.

* **Stage 2 (Diagnosis):**
  A multi-class classifier trained **only on anomalous samples** to distinguish fault types without interference from dominant normal patterns.

This hierarchical design explicitly decouples detection and diagnosis objectives, mitigating optimization conflicts under severe class imbalance.

---

## Architecture Details (Common)

### LSTM Autoencoder

* Window length (sequence length): 5
* Sliding window stride: 1 (80% overlap)
* Encoder–decoder: single-layer LSTM
* Reconstruction loss: Mean Squared Error (MSE)

### Diagnostic Classifier

* Input: latent representation from the LSTM encoder (last time step)
* Architecture: lightweight MLP (single hidden layer)
* Loss: weighted cross-entropy

---

## Dataset-specific Best Hyperparameters

The optimal hyperparameters were independently selected for each dataset based on validation Macro F1-score.

| Hyperparameter             | Dataset 4 | Dataset 5 | Dataset 6 |
| -------------------------- | --------- | --------- | --------- |
| Hidden size                | 16        | 32        | 16        |
| Sequence length            | 5         | 5         | 5         |
| Learning rate              | 0.0005    | 0.0001    | 0.0005    |
| Batch size                 | 4         | 4         | 4         |
| Reconstruction loss weight | 1         | 1         | 1         |
| Anomaly loss weight        | 5         | 5         | 10        |
| Class loss weight          | 10        | 15        | 15        |
| Anomaly threshold (τ)      | 0.5       | 0.5       | 0.5       |

---

## Training Configuration

* Optimizer: Adam
* Maximum epochs: 1000
* Early stopping: enabled (patience = 10)
* Loss weighting: enabled to address severe class imbalance
* Regularization: no explicit dropout or L2 regularization applied

---

## Computational Information

| Hidden Size | Trainable Parameters | Model Size |
| ----------: | -------------------: | ---------: |
|          16 |                4,773 |   0.018 MB |
|          32 |               12,101 |   0.046 MB |

Training time and inference efficiency vary depending on dataset size and operating conditions and are reported in detail in the experimental section of the paper.

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
```

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

---

## Notes

* The anomaly threshold τ is fixed to 0.5 based on sensitivity analysis showing consistent optimal performance across datasets.
* During inference, only sequences identified as anomalous are forwarded to the diagnostic classifier, reducing unnecessary computation and supporting real-time deployment.

---
