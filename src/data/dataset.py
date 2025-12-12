import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class NNDLDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # csv file is for annotations
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        # labels
        superclass = self.data_frame.iloc[idx, 1]
        subclass = self.data_frame.iloc[idx, 2]
        
        # transform
        if self.transform:
            image = self.transform(image)

        return image, superclass, subclass
