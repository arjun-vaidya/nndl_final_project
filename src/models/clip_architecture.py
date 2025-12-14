import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, CLIPConfig
from src.utils.constants import CLIP_MODEL_NAME

class CustomCLIP(nn.Module):
    def __init__(self, model_name = CLIP_MODEL_NAME, pretrained=True):
        super(CustomCLIP, self).__init__()
        
        if pretrained:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            config = CLIPConfig()
            self.model = CLIPModel(config)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def forward(self, pixel_values, input_ids, attention_mask=None):

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=True
        )
        return outputs.loss, outputs.logits_per_image

    def encode_image(self, pixel_values):
        # normalizes the image embeddings
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def encode_text(self, input_ids, attention_mask=None):
        # normalizes the text embeddings
        text_features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)