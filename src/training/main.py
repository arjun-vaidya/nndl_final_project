
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from src.data.dataset import NNDLDataset
from src.data.transforms import get_transforms
from src.models.resnet_dual_head import DualHeadResNet

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    full_dataset = NNDLDataset(
        csv_file=os.path.join(args.data_dir, 'train_data.csv'),
        img_dir=os.path.join(args.data_dir, 'train_images'),
        transform=get_transforms('train')
    )
    
    # 2-Way Split: Train / Val
    total_len = len(full_dataset)
    val_len = int(args.val_split * total_len)
    train_len = total_len - val_len
    
    print(f"Dataset Split: Train={train_len}, Val={val_len}")
    
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model Setup
    model = DualHeadResNet(num_superclasses=3, num_subclasses=87, pretrained=True)
    model = model.to(device)

    # Criterions
    criterion_super = nn.CrossEntropyLoss()
    criterion_sub = nn.CrossEntropyLoss()

    # STAGE 1: Train Heads Only
    print("\n--- STAGE 1: Training Heads Only ---")
    model.freeze_backbone()
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_stage1)
    
    for epoch in range(args.epochs_stage1):
        loss, super_acc, sub_acc = model.train_epoch(train_loader, optimizer, criterion_super, criterion_sub, device)
        val_loss, val_super_acc, val_sub_acc = model.validate_epoch(val_loader, criterion_super, criterion_sub, device)
        
        print(f"Epoch {epoch+1}/{args.epochs_stage1} [Stage 1] | Loss: {loss:.4f} | "
              f"Train Acc: Super={super_acc:.2f}% Sub={sub_acc:.2f}% | "
              f"Val Acc: Super={val_super_acc:.2f}% Sub={val_sub_acc:.2f}%")
        
        if args.save_checkpoints:
             ckpt_path = f"checkpoints/stage1_epoch{epoch+1}.pth"
             torch.save(model.state_dict(), ckpt_path)
             print(f"Saved checkpoint: {ckpt_path}")

    # STAGE 2: Finetune All
    print("\n--- STAGE 2: Fine-tuning Whole Model ---")
    model.unfreeze_backbone()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_stage2)
    
    for epoch in range(args.epochs_stage2):
        loss, super_acc, sub_acc = model.train_epoch(train_loader, optimizer, criterion_super, criterion_sub, device)
        val_loss, val_super_acc, val_sub_acc = model.validate_epoch(val_loader, criterion_super, criterion_sub, device)
        
        print(f"Epoch {epoch+1}/{args.epochs_stage2} [Stage 2] | Loss: {loss:.4f} | "
              f"Train Acc: Super={super_acc:.2f}% Sub={sub_acc:.2f}% | "
              f"Val Acc: Super={val_super_acc:.2f}% Sub={val_sub_acc:.2f}%")

        if args.save_checkpoints:
             ckpt_path = f"checkpoints/stage2_epoch{epoch+1}.pth"
             torch.save(model.state_dict(), ckpt_path)
             print(f"Saved checkpoint: {ckpt_path}")

    # Save Model
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/resnet_dual_head_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hierarchical ResNet')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs_stage1', type=int, default=3, help='Epochs for Stage 1 (Heads only)')
    parser.add_argument('--epochs_stage2', type=int, default=10, help='Epochs for Stage 2 (Finetuning)')
    parser.add_argument('--lr_stage1', type=float, default=1e-3, help='Learning rate for Stage 1')
    parser.add_argument('--lr_stage2', type=float, default=1e-4, help='Learning rate for Stage 2')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--save_checkpoints', type=int, default=1, help='Save checkpoints: 1=Yes, 0=No')
    
    args = parser.parse_args()
    main(args)