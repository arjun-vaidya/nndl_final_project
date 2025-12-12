import torch
from torchvision import transforms

# img net stats
# this normalizes the image with mean and std dev 1
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

#split (str): 'train' or 'val'/'test'
def get_transforms(split='train'):
    
    if split == 'train':
        return transforms.Compose([
            # adding noise
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),

            # normalizing
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        # no noise
        # but still need to normalize
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
