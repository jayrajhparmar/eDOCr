import os
import pandas as pd
import shutil
import random

def clean_and_split_dataset(input_dir, output_dir, max_file_size_kb=100, split_ratio=0.8):
    print(f"Cleaning dataset from {input_dir}...")
    
    csv_path = os.path.join(input_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    valid_samples = []
    
    # 1. Filter out garbage and overly large images
    for index, row in df.iterrows():
        filename = str(row['filename']).strip()
        text = str(row['text']).strip()
        
        # Skip empty text
        if not text or text.lower() == 'nan':
            continue
            
        img_path = os.path.join(input_dir, "images", filename)
        if not os.path.exists(img_path):
            continue
            
        # Check file size (in bytes)
        file_size = os.path.getsize(img_path)
        if file_size > (max_file_size_kb * 1024):
            print(f"Skipping {filename}: Size {file_size/1024:.2f}KB is over the {max_file_size_kb}KB limit.")
            continue
            
        valid_samples.append((filename, text, img_path))
        
    print(f"\nFound {len(valid_samples)} valid crops out of {len(df)} original rows.")
    
    if len(valid_samples) == 0:
        print("No valid samples found!")
        return
        
    # 2. Shuffle and Split
    random.shuffle(valid_samples)
    split_index = int(len(valid_samples) * split_ratio)
    
    train_samples = valid_samples[:split_index]
    eval_samples = valid_samples[split_index:]
    
    # 3. Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in [("train", train_samples), ("eval", eval_samples)]:
        split_dir = os.path.join(output_dir, split_name)
        split_img_dir = os.path.join(split_dir, "images")
        os.makedirs(split_img_dir, exist_ok=True)
        
        split_csv_path = os.path.join(split_dir, "labels.csv")
        
        with open(split_csv_path, "w", encoding="utf-8") as f:
            f.write("filename,text\n")
            for filename, text, img_path in split_data:
                # Copy image
                dst_img_path = os.path.join(split_img_dir, filename)
                shutil.copy2(img_path, dst_img_path)
                
                # Write to csv
                f.write(f"{filename},{text}\n")
                
        print(f"Saved {len(split_data)} samples to {split_dir}")

if __name__ == "__main__":
    input_directory = r"D:\eOCR\full_training_dataset"
    output_directory = r"D:\eOCR\cleaned_dataset"
    
    # 100KB is generous for a text crop. 
    # Anything larger is usually a massive hallucination block.
    clean_and_split_dataset(input_directory, output_directory, max_file_size_kb=100)
    print("Dataset cleanup complete!")
