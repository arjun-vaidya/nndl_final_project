import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

def check_images(img_dir, csv_file):
    df = pd.read_csv(csv_file)
    corrupt_files = []
    
    print(f"FILE INTEGRITY CHECK: Checking {len(df)} images in {img_dir}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_dir, row['image'])
        try:
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f'Bad file: {img_path}')
            corrupt_files.append(row['image'])
            
    if not corrupt_files:
        print("FILE INTEGRITY CHECK: All images verified successfully!")
    else:
        print(f"FILE INTEGRITY CHECK: Found {len(corrupt_files)} corrupt images.")
        
    return corrupt_files

if __name__ == "__main__":
    check_images('data/train_images', 'data/train_data.csv')
