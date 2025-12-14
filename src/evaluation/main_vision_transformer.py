import torch
import os
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from src.models.vision_transformer_architecture import CrossAttentionViT
from src.data.transforms import get_transforms

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(img_dir)
            # hidden files start with this
            if not f.startswith('.')
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}")

    # not pretrained (loading weights)
    model = CrossAttentionViT(num_superclasses=3, num_subclasses=87, pretrained = False)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # data loader
    print(f"Loading test images from {args.data_dir}")

    # need to upscale for ViT
    transform = get_transforms('valid', upscale=True)
    test_dataset = TestDataset(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = []

    print("Starting inference")
    with torch.no_grad():
        for images, filenames in tqdm(test_loader):
            images = images.to(device)
            
            preds = model.predict(images, threshold=args.threshold)
            
            super_preds = preds['superclass'].cpu().numpy()
            sub_preds = preds['subclass'].cpu().numpy()
            
            for filename, super_p, sub_p in zip(filenames, super_preds, sub_preds):
                results.append({
                    'image': filename,
                    'superclass_index': super_p,
                    'subclass_index': sub_p
                })

    # csv
    df = pd.DataFrame(results)
    df = df[['image', 'superclass_index', 'subclass_index']]
    
    # sorting by image filename (integer part)
    df['sort_key'] = df['image'].apply(lambda x: int(os.path.splitext(x)[0]))
    df.sort_values('sort_key', inplace=True)
    df.drop('sort_key', axis=1, inplace=True)
    
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer on Test Data')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pth file')
    parser.add_argument('--output', type=str, default='test_predictions.csv', help='Output CSV filename')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold for Novel class detection')
    
    args = parser.parse_args()
    evaluate(args)