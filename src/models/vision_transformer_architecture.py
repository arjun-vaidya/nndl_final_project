
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import numpy as np

class CrossAttentionViT(nn.Module):
    def __init__(self, num_superclasses=3, num_subclasses=87, pretrained=True):
        super(CrossAttentionViT, self).__init__()
        
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights)
        
        # feature dimension for ViT is 768
        self.feature_dim = 768
        self.proj_dim = 512
        
        # remove head
        self.backbone.heads = nn.Identity()
        
        # super class projections
        self.super_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # sub class projections
        self.sub_proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # cross-attention
        # goal is to capture dependencies 
        # between base and super class
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        self.norm_sub = nn.LayerNorm(self.proj_dim)
        
        # classification heads (1 more for novel)
        self.super_head = nn.Linear(self.proj_dim, num_superclasses + 1)
        self.sub_head = nn.Linear(self.proj_dim, num_subclasses + 1)
        
    def forward(self, x):
        # 768 dimensions
        features = self.backbone(x)
        
        # 512 dimensions
        super_feat = self.super_proj(features) 
        sub_feat = self.sub_proj(features)     
        
        # cross-attention
        # [Batch, 1, 512]
        # NOTE: sequence of length 1
        query = sub_feat.unsqueeze(1) 
        key = super_feat.unsqueeze(1)
        value = super_feat.unsqueeze(1)
        
        attn_output, _ = self.cross_attention(query, key, value)
        
        # attention only for sub class features
        sub_feat_refined = self.norm_sub(sub_feat + attn_output.squeeze(1))
        
        # final predictions
        super_logits = self.super_head(super_feat)
        sub_logits = self.sub_head(sub_feat_refined)
        
        return super_logits, sub_logits

    def predict(self, x, threshold=0.7, mapping_consistency_check=None):
        self.eval()

        with torch.no_grad():
            super_logits, sub_logits = self.forward(x)
            
            # softmax
            sub_probs = torch.softmax(sub_logits, dim = 1)
            super_probs = torch.softmax(super_logits, dim = 1)
            
            # predictions
            max_sub_probs, sub_preds = torch.max(sub_probs, dim = 1)
            max_super_probs, super_preds = torch.max(super_probs, dim = 1)
            
            # thresholding
            mask_novel_super = max_super_probs < threshold
            super_preds[mask_novel_super] = 3
            
            mask_novel_sub = max_sub_probs < threshold
            sub_preds[mask_novel_sub] = 87
            
            # hierarchy enforcement
            # if either is novel, other is also novel
            sub_preds[mask_novel_super] = 87
            super_preds[mask_novel_sub] = 3

            # consistency check
            final_sub_preds = sub_preds.clone()
            
            if mapping_consistency_check:
                for i in range(len(x)):
                    # already novel
                    if final_sub_preds[i] == 87: 
                        continue
                    if super_preds[i] == 3: 
                        continue
                    
                    s_pred = super_preds[i].item()
                    sub_pred = sub_preds[i].item()
                    
                    if not mapping_consistency_check(s_pred, sub_pred):
                        final_sub_preds[i] = 87
                        super_preds[i] = 3

            return {
                'superclass': super_preds,
                'subclass': final_sub_preds,
                'super_probs': max_super_probs,
                'sub_probs': max_sub_probs
            }

    def train_epoch(self, loader, optimizer, criterion_super, criterion_sub, device):
        self.train()
        
        running_loss = 0.0
        correct_super = 0
        correct_sub = 0
        total = 0

        loop = tqdm(loader, desc = "Training Vision Transformer")
        
        for images, super_labels, sub_labels in loop:
            images = images.to(device)
            super_labels = super_labels.to(device)
            sub_labels = sub_labels.to(device)

            # forward
            super_logits, sub_logits = self.forward(images)

            # loss
            loss_super = criterion_super(super_logits, super_labels)
            loss_sub = criterion_sub(sub_logits, sub_labels)
            loss = loss_super + loss_sub

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item()
            _, predicted_super = torch.max(super_logits.data, 1)
            _, predicted_sub = torch.max(sub_logits.data, 1)
            
            total += super_labels.size(0)
            correct_super += (predicted_super == super_labels).sum().item()
            correct_sub += (predicted_sub == sub_labels).sum().item()
            
            # adding it to progress bar (tqdm)
            loop.set_postfix(loss = loss.item())

        epoch_loss = running_loss / len(loader)
        super_acc = 100 * correct_super / total
        sub_acc = 100 * correct_sub / total
        
        return epoch_loss, super_acc, sub_acc

    def validate_epoch(self, loader, criterion_super, criterion_sub, device):
        self.eval()

        running_loss = 0.0
        correct_super = 0
        correct_sub = 0
        total = 0

        with torch.no_grad():

            for images, super_labels, sub_labels in loader:
                images = images.to(device)
                super_labels = super_labels.to(device)
                sub_labels = sub_labels.to(device)

                super_logits, sub_logits = self.forward(images)

                loss_super = criterion_super(super_logits, super_labels)
                loss_sub = criterion_sub(sub_logits, sub_labels)
                loss = loss_super + loss_sub

                running_loss += loss.item()

                _, predicted_super = torch.max(super_logits.data, 1)
                _, predicted_sub = torch.max(sub_logits.data, 1)

                total += super_labels.size(0)
                correct_super += (predicted_super == super_labels).sum().item()
                correct_sub += (predicted_sub == sub_labels).sum().item()

        avg_loss = running_loss / len(loader)
        super_acc = 100 * correct_super / total
        sub_acc = 100 * correct_sub / total
        
        return avg_loss, super_acc, sub_acc

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True