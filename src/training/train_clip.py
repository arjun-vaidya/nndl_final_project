import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from src.models.clip_architecture import CustomCLIP
from src.data.dataset_clip import CLIPDataset
from src.utils.constants import CLIP_MODEL_NAME

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Using CLIP model: {args.model_name}")
    custom_clip = CustomCLIP(model_name=args.model_name, pretrained=True).to(device)
    
    print(f"Loading data from {args.csv_file}")
    dataset = CLIPDataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        processor=custom_clip.processor,
        # only for stage 2
        augment_rotation=args.stage == 2 
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # optimizer
    # ensuring low LR for second step
    lr = args.lr if args.stage == 1 else args.lr / 10
    # also adding weight decay
    optimizer = optim.AdamW(custom_clip.parameters(), lr=lr, weight_decay = args.weight_decay)

    # training loop
    custom_clip.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # data
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            loss, _ = custom_clip(pixel_values, input_ids, attention_mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # save checkpoint
        save_path = os.path.join(args.checkpoint_dir, f"clip_stage{args.stage}_epoch{epoch+1}")
        custom_clip.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune CLIP')
    parser.add_argument('--csv_file', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default=CLIP_MODEL_NAME)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Stage 1: Standard, Stage 2: Rotated')
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
