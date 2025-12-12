import torch
from torch.utils.data import Dataset, ConcatDataset
import random
from torchvision import transforms
from PIL import Image, ImageOps 
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from src.utils.constants import MEAN, STD

# create novel data (Super = 3, Sub = 87)
class NovelDataset(Dataset):

    def __init__(self, original_dataset, num_samples, transform=None):
        self.original_dataset = original_dataset
        self.num_samples = num_samples
        self.transform = transform
        self.indices = np.random.choice(len(original_dataset), num_samples, replace=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # first animal
        idx1 = self.indices[idx]
        img1, _, _ = self.get_unnormalized_pil(idx1)
        
        # second animal
        idx2 = np.random.randint(len(self.original_dataset))
        img2, _, _ = self.get_unnormalized_pil(idx2)
        
        # blend or distort
        if random.random() > 0.5:
             novel_img = Image.blend(img1, img2, alpha=0.5)
        else:
             novel_img = self.apply_color_distortion(img1)
        
        if self.transform:
            novel_img = self.transform(novel_img)
            
        # novel labels
        return novel_img, 3, 87

    def get_unnormalized_pil(self, idx):
        img, lbl_super, lbl_sub = self.original_dataset[idx]
        
        if isinstance(img, torch.Tensor):
            # Un-normalize
            mean = torch.tensor(MEAN).view(3, 1, 1)
            std = torch.tensor(STD).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            to_pil = transforms.ToPILImage()
            img = to_pil(img)
            
        return img, lbl_super, lbl_sub

    def apply_color_distortion(self, img):
        img = self.shuffle_channels(img)
        
        if random.random() > 0.5:
            img = ImageOps.invert(img)
        else:
            img = ImageOps.solarize(img, threshold=100)
        return img

    def shuffle_channels(self, img):
        bands = list(img.split())
        random.shuffle(bands)
        return Image.merge('RGB', bands)

def append_novel_dataset(train_dataset, fraction=0.1, transform=None):
    num_novel = int(len(train_dataset) * fraction)
    print(f"Adding {num_novel} novel samples to training data")
    
    novel_ds = NovelDataset(
        train_dataset, 
        num_samples=num_novel, 
        transform=transform
    )
    
    combined_ds = ConcatDataset([train_dataset, novel_ds])
    return combined_ds


def show_images_grid(dataset, num_images=64, nrow=8):
    images = []
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for idx in indices:
        img, _, _ = dataset[idx]
        # un normalize
        if isinstance(img, torch.Tensor):
             img = img * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
             img = torch.clamp(img, 0, 1)
        images.append(img)
    
    # grid
    grid_img = torchvision.utils.make_grid(images, nrow=nrow)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    # plt.show()
    
    # saving
    plt.savefig("novel_grid.png")
    print("Saved grid sample to novel_grid.png")