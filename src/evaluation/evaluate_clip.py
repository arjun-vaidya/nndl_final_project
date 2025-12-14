import torch
import pandas as pd
from PIL import Image
import os
import argparse
from tqdm import tqdm
from src.models.clip_architecture import CustomCLIP
import torch.nn.functional as F
from src.utils.constants import CLIP_MODEL_NAME, NUM_SUBCLASSES, NUM_SUPERCLASSES, CLIP_MAX_CONTEXT_LENGTH

def build_vector_representations_per_class(model, data_file, device):
    # NOTE: this returns map of class index (subclass)
    # to vector representation (averaged over all descriptions)
    df = pd.read_csv(data_file)
    class_vector_map = {}
    unique_classes = NUM_SUBCLASSES
    
    model.model.eval()
    with torch.no_grad():
        for cls_idx in tqdm(range(unique_classes), desc="Subclasses"):
            descriptions = df[df['subclass_index'] == cls_idx]['description'].tolist()
            unique_descriptions = list(set(descriptions))
            
            inputs = model.processor(text=unique_descriptions, return_tensors="pt", padding=True, truncation=True, max_length=CLIP_MAX_CONTEXT_LENGTH).to(device)
            text_features = model.encode_text(inputs['input_ids'], inputs['attention_mask'])

            # computing the mean for the representations
            class_representation = text_features.mean(dim=0, keepdim=True)
            class_representation = class_representation / class_representation.norm(p=2, dim=-1, keepdim=True)
            class_vector_map[cls_idx] = class_representation

    return class_vector_map

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.checkpoint:
        print(f"Loading fine-tuned model form {args.checkpoint}")
        model = CustomCLIP(model_name=args.checkpoint, pretrained=True).to(device)
    else:
        print(f"[VERIFY TEST] Loading base model {args.model_name}")
        model = CustomCLIP(model_name=args.model_name, pretrained=True).to(device)
    
    # building vector representations for each class
    class_vector_map = build_vector_representations_per_class(model, args.train_csv, device)
    sorted_class_indices = sorted(class_vector_map.keys())
    
    # stacking them
    reference_embeddings = torch.cat([class_vector_map[i] for i in sorted_class_indices], dim=0) # (87, dim)
    print(f"reference embeddings shape (stacked): {reference_embeddings.shape}")

    # running inference
    test_files = [f for f in os.listdir(args.test_dir) if not f.startswith('.')]
    results = []

    # loading train csv for mapping
    df_train = pd.read_csv(args.train_csv)
    subclass_to_superclass = dict(zip(df_train['subclass_index'], df_train['superclass_index']))

    print(f"evaluating {len(test_files)} test images")
    
    model.model.eval()
    with torch.no_grad():
        for img_file in tqdm(test_files):
            img_path = os.path.join(args.test_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            
            inputs = model.processor(images=image, return_tensors="pt").to(device)
            image_feature = model.encode_image(inputs['pixel_values']) 
            
            # computing dot product
            dot_products = (image_feature @ reference_embeddings.T).squeeze(0)
            max_score, best_class_idx = dot_products.max(dim=0)
            
            # most similar class
            predicted_subclass = sorted_class_indices[best_class_idx.item()]
            
            # if score is below threshold, it is Novel
            if max_score.item() < args.threshold:
                predicted_subclass = NUM_SUBCLASSES 
                predicted_superclass = NUM_SUPERCLASSES 
            else:
                # subclass mappend to superclass
                predicted_superclass = subclass_to_superclass.get(predicted_subclass, 0) # Default to 0 if issue, but shouldn't happen

            results.append({
                'image': img_file,
                'superclass_index': predicted_superclass,
                'subclass_index': predicted_subclass,
                'confidence': max_score.item()
            })

    # saving results (similar logic)

    # sorting by image filename (integer part)
    df_res = pd.DataFrame(results)
    df_res['sort_key'] = df_res['image'].apply(lambda x: int(os.path.splitext(x)[0]))
    df_res.sort_values('sort_key', inplace=True)
    df_res.drop('sort_key', axis=1, inplace=True)
    
    df_res[['image', 'superclass_index', 'subclass_index']].to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

    # results with confidence
    confidence_output = 'visualize_confidence.csv'
    df_res[['image', 'superclass_index', 'subclass_index', 'confidence']].to_csv(confidence_output, index=False)
    print(f"Saved csv with confidence to {confidence_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CLIP')
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--train_csv', type=str, required=True, help='Path to Train CSV for generating reference embeddings')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to fine-tuned model dir')
    parser.add_argument('--model_name', type=str, default=CLIP_MODEL_NAME)
    parser.add_argument('--threshold', type=float, default=0.25, help='Cosine similarity threshold for Novel classes')
    parser.add_argument('--output', type=str, default='prediction_clip.csv')
    
    args = parser.parse_args()
    evaluate(args)
