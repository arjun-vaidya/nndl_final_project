from torchvision import transforms
from src.utils.constants import MEAN, STD

# train or (val or test)
def get_transforms(split='train', upscale=False):
    
    transforms_list = []
    
    # upscaling for vision transformer
    if upscale:
        transforms_list.append(transforms.Resize((224, 224)))
    
    if split == 'train':
        transforms_list.extend([
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
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        
    return transforms.Compose(transforms_list)