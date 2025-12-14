# ImageNet normalization statistics
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Data Augmentation
NUM_ROTATIONS = 10

# Paths
DATA_DIR = 'data'
TRAIN_IMG_DIR = 'data/train_images'
AUGMENTED_IMG_DIR = 'data/augmented_train_images'

# Classes
NUM_SUPERCLASSES = 3
NUM_SUBCLASSES = 87

# CLIP
CLIP_MAX_CONTEXT_LENGTH = 77
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'