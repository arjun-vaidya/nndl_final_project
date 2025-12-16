This project implements transfer learning on 3 major architectures (5 approaches).

| Model Name | Super Acc. | Seen Super Acc. | Unseen Super Acc. | Sub Acc. | Seen Sub Acc. | Unseen Sub Acc. | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet (Version 1)** | 68.25% | 66.55% | 72.56% | 59.36% | 83.30% | 52.90% | Baseline ResNet |
| **ViT (Approach 1)** | 71.06% | 82.78% | 41.33% | 42.88% | 93.35% | 29.25% | Stage 2 Ep 10 |
| **ViT (Approach 2)** | 67.36% | 74.63% | 48.92% | 49.14% | 89.61% | 38.21% | Stage 1 Ep 10 |
| **CLIP (Approach 1)** | 83.26% | 99.41% | 42.28% | 32.36% | 96.00% | 15.18% | Threshold 0.25 |
| **CLIP (Approach 2)** | 92.12% | 93.58% | 88.42% | 49.68% | 95.96% | 37.18% | Threshold 0.30 |

## NOTE

1. Create a `.env` file in the root directory with the following template:

```bash
INSTANCE_NAME=""
ZONE=""
REMOTE_DIR=""
```

2. Create a `data` directory with the provided data.