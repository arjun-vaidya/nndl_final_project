# NNDL Final Project: Image Classification

The project is structured as follows 

```
.
├── checkpoints/       # Saved model weights
├── data/              # Dataset files
│   ├── train_images/  # Training images
│   ├── test_images/   # Test images
│   └── *.csv          # Metadata files (including prediction.csv)
├── src/               # Source code
│   ├── data/          # Data loading and preprocessing
│   │   ├── cleaning.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── evaluation/    # Evaluation and inference scripts
│   │   ├── main.py
│   ├── models/        # Model definitions
│   │   ├── resnet_dual_head.py
│   │   └── vision_transformer_architecture.py
│   ├── training/      # Training scripts
│   │   └── main.py
│   └── utils/         # Utility functions
│       └── mapping.py
└── prediction.csv     # Model output

```

---

### 1. Training ResNet Dual Head
The training pipeline runs in two stages:
1.  **Stage 1**: Freezes the backbone and trains the classification heads.
2.  **Stage 2**: Unfreezes the backbone for fine-tuning.

```bash
python3 -m src.training.main \
  --data_dir data \
  --epochs_stage1 3 \
  --epochs_stage2 10 \
  --batch_size 32
```

### 2. Evaluation ResNet Dual Head
Generate predictions for the test set.

```bash
python3 -m src.evaluation.main \
  --data_dir data/test_images \
  --checkpoint checkpoints/resnet_dual_head_final.pth \
  --output prediction.csv
```

---

## Model Performance

| Model Name | Super Acc. | Seen Super Acc. | Unseen Super Acc. | Sub Acc. | Seen Sub Acc. | Unseen Sub Acc. | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet (Version 1)** | 68.25% | 66.55% | 72.56% | 59.36% | 83.30% | 52.90% | Baseline ResNet |
| **ViT (Approach 1)** | 71.06% | 82.78% | 41.33% | 42.88% | 93.35% | 29.25% | Stage 2 Ep 10 |
| **ViT (Approach 2)** | 67.36% | 74.63% | 48.92% | 49.14% | 89.61% | 38.21% | Stage 1 Ep 10 |
| **CLIP (Approach 1)** | 83.26% | 99.41% | 42.28% | 32.36% | 96.00% | 15.18% | Threshold 0.25 |
| **CLIP (Approach 2)** | 92.12% | 93.58% | 88.42% | 49.68% | 95.96% | 37.18% | Threshold 0.30 |