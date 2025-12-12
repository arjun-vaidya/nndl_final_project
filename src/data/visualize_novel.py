
import os
import torch
from src.data.dataset import NNDLDataset
from src.data.transforms import get_transforms
from src.data.novelty_generation import append_novel_dataset, show_images_grid
from src.data.novelty_generation import NovelDataset

def main():
    data_dir = 'data'

    print("Loading original dataset")
    train_ds = NNDLDataset(
        csv_file=os.path.join(data_dir, 'train_data.csv'),
        img_dir=os.path.join(data_dir, 'train_images'),
        transform=get_transforms('train') 
    )
    
    print(f"Original dataset size: {len(train_ds)}")
    
    num_samples = 64
    print(f"Generating {num_samples} novel samples")
    
    novel_ds = NovelDataset(
        train_ds, 
        num_samples=num_samples, 
        transform=get_transforms('train')
    )
    
    print("Generating 8x8 grid novel samples")
    
    show_images_grid(novel_ds, num_images=64, nrow=8)

if __name__ == '__main__':
    main()
