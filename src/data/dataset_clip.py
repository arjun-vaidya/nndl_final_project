import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from src.utils.constants import CLIP_MAX_CONTEXT_LENGTH

class CLIPDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, augment_rotation=False, num_rotations=10):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.augment_rotation = augment_rotation
        self.num_rotations = num_rotations if augment_rotation else 1
        
        # same logic as create_training
        if self.augment_rotation:
            self.angles = np.linspace(0, 360, num_rotations, endpoint=False)
        else:
            self.angles = [0.0]

    def __len__(self):
        return len(self.data) * self.num_rotations

    def __getitem__(self, idx):
        # adding rotations
        original_idx = idx // self.num_rotations
        rotation_idx = idx % self.num_rotations
        
        row = self.data.iloc[original_idx]
        img_name = row['image']
        description = row['description']
        
        # image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # applying rotations
        angle = self.angles[rotation_idx]
        if angle > 0:
            image = image.rotate(angle, expand=False)
            description = f"Rotated view of {description}"

        inputs = self.processor(
            text=[description], 
            images=image, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=CLIP_MAX_CONTEXT_LENGTH 
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }