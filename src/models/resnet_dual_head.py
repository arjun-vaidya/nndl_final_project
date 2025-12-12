
import torch
import torch.nn as nn
from torchvision import models

class DualHeadResNet(nn.Module):

    # num_superclasses: superclass outputs (3: bird, dog, reptile).
    # num_subclasses: subclass outputs (87).
    # pretrained: use ImageNet pre-trained weights.

    def __init__(self, num_superclasses=3, num_subclasses=87, pretrained=True):
        super(DualHeadResNet, self).__init__()
        
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # dimension before the final layer (512)
        num_features = self.backbone.fc.in_features
        
        # removing original classification head
        self.backbone.fc = nn.Identity()
        
        # separate heads for superclass and subclass prediction
        self.super_head = nn.Linear(num_features, num_superclasses)
        self.sub_head = nn.Linear(num_features, num_subclasses)
        
    def forward(self, x):
        # features from backbone
        features = self.backbone(x)
        
        # logits for new heads
        super_logits = self.super_head(features)
        sub_logits = self.sub_head(features)
        
        return super_logits, sub_logits
    
    def predict(self, x, threshold=0.7, mapping_consistency_check=None):

        self.eval()

        with torch.no_grad():
            super_logits, sub_logits = self.forward(x)
            
            # softmax
            sub_probs = torch.softmax(sub_logits, dim=1)
            super_probs = torch.softmax(super_logits, dim=1)
            
            # predictions and confidences

            # subclass predictions is the index of the max probability
            max_sub_probs, sub_preds = torch.max(sub_probs, dim=1)
            # superclass predictions is the index of the max probability
            max_super_probs, super_preds = torch.max(super_probs, dim=1)
            
            # 1. Superclass Confidence
            # If confidence < threshold, predict Novel (Index -1)
            mask_novel_super = max_super_probs < threshold
            super_preds[mask_novel_super] = -1
            
            # 2. Subclass Confidence
            # If subclass confidence < threshold, predict Novel (Index -1)
            mask_novel_sub = max_sub_probs < threshold
            sub_preds[mask_novel_sub] = -1
            
            # if either is Novel, both are Novel
            sub_preds[mask_novel_super] = -1
            super_preds[mask_novel_sub] = -1

            # Create a copy of sub_preds to potentially modify based on consistency check
            final_sub_preds = sub_preds.clone()

            # 3. Hierarchy Consistency
            # superclass and subclass do not match
            # it is likely a Novel class or confusion
            if mapping_consistency_check:
                for i in range(len(x)):
                    # novel subclass
                    if final_sub_preds[i] == -1:
                        continue

                    # superclass and subclass values    
                    super_pred_item = super_preds[i].item()
                    sub_pred_item = sub_preds[i].item() 

                    # novel superclass
                    if super_pred_item == -1:
                         continue
                         
                    # checking if sub_pred is a valid child of super_pred
                    is_consistent = mapping_consistency_check(super_pred_item, sub_pred_item)
                    
                    if not is_consistent:
                        final_sub_preds[i] = -1

            return {
                'superclass': super_preds,
                'subclass': final_sub_preds,
                'super_probs': max_super_probs,
                'sub_probs': max_sub_probs
            }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Frozen ResNet18 layers (not the heads)")
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Unfrozen ResNet18 layers")
