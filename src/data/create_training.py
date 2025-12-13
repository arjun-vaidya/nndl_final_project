
import os
import io
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.utils.constants import NUM_ROTATIONS, DATA_DIR, TRAIN_IMG_DIR, AUGMENTED_IMG_DIR
import torchvision.transforms.functional as TF

def main():
    os.makedirs(AUGMENTED_IMG_DIR, exist_ok = True)
    
    # getting labels
    train_csv = os.path.join(DATA_DIR, 'train_data.csv')
    df = pd.read_csv(train_csv)
    
    new_data = []
    
    print(f"Generating {NUM_ROTATIONS} rotations for each training image")
    
    # angles 360 / NUM_ROTATES (10) = 36 degrees step
    angles = np.linspace(0, 360, NUM_ROTATIONS, endpoint = False)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(row['image'])
        
        if not img_id.endswith('.jpg'):
            img_id += '.jpg'
            
        src_path = os.path.join(TRAIN_IMG_DIR, img_id)
        
        try:
            img = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f"error at {src_path}: {e}")
            continue
            
        base_name = os.path.splitext(img_id)[0]
        
        for i, angle in enumerate(angles):

            # rotate
            rotated_img = img.rotate(angle, expand=False)        
            new_filename = f"{base_name}_rot{i}.jpg"

            # save new image
            save_path = os.path.join(AUGMENTED_IMG_DIR, new_filename)
            rotated_img.save(save_path)
            
            # add to new data
            new_data.append({
                'image': new_filename,
                'superclass_index': row['superclass_index'],
                'subclass_index': row['subclass_index']
            })
            
    # save new CSV
    new_df = pd.DataFrame(new_data)
    new_csv_path = os.path.join(DATA_DIR, 'augmented_train_data.csv')
    new_df.to_csv(new_csv_path, index=False)

    print(f"Augmented dataset info saved to {new_csv_path}")
    print(f"Total images: {len(new_df)}")

if __name__ == '__main__':
    main()
