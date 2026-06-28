import os
import glob
import csv
import random
import cv2
import fitz  # PyMuPDF supports TIFF natively as well!
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from eDOCr import tools
import torch
from pathlib import Path

# Set huggingface cache AND temp directories to D drive since C drive is completely full
os.environ["HF_HOME"] = r"D:\eOCR\.cache"
os.environ["TMPDIR"] = r"D:\eOCR\tmp"
os.environ["TEMP"] = r"D:\eOCR\tmp"
os.environ["TMP"] = r"D:\eOCR\tmp"
os.makedirs(r"D:\eOCR\tmp", exist_ok=True)

def auto_label_dataset(file_path, processor, model, device, output_dir="full_training_dataset"):
    print(f"\nGenerating dataset from {os.path.basename(file_path)}...")
    
    img_dir = os.path.join(output_dir, "images")
    holdout_img_dir = os.path.join(output_dir, "holdout_images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(holdout_img_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "labels.csv")
    holdout_csv_path = os.path.join(output_dir, "holdout_labels.csv")
    
    for path in [csv_path, holdout_csv_path]:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "text"])
            
    try:
        # PyMuPDF can open both PDFs and TIFFs
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Failed to open {file_path} with PyMuPDF: {e}")
        return

    for i in range(len(doc)):
        print(f"Processing page {i+1}/{len(doc)}...")
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        print("Extracting bounding boxes...")
        try:
            class_list, img_boxes = tools.box_tree.findrect(img)
            boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
        except Exception as e:
            print(f"CV2 extraction failed: {e}")
            continue
            
        raw_lists = [class_list, gdt_boxes, boxes_infoblock]
        all_boxes = []
        for lst in raw_lists:
            if isinstance(lst, list):
                for item in lst:
                    if hasattr(item, 'crop_img'):
                        all_boxes.append(item)
                    elif isinstance(item, list):
                        for sub_item in item:
                            if hasattr(sub_item, 'crop_img'):
                                all_boxes.append(sub_item)
            elif hasattr(lst, 'crop_img'):
                all_boxes.append(lst)
        
        print(f"Found {len(all_boxes)} text regions.")
        
        with open(csv_path, "a", encoding="utf-8") as f:
            for j, box in enumerate(all_boxes):
                if box.crop_img is None or box.crop_img.shape[0] == 0 or box.crop_img.shape[1] == 0:
                    continue
                
                height, width = box.crop_img.shape[:2]
                if height > 1000 or width > 2000:
                    continue 
                
                img_rgb = cv2.cvtColor(box.crop_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                try:
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                    generated_ids = model.generate(pixel_values, max_new_tokens=30)
                    text_guess = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                except Exception as e:
                    text_guess = ""
                
                text_guess = text_guess.replace("\r", " ").replace("\n", " ").strip()
                if text_guess != "":
                    clean_name = "".join([c if c.isalnum() else "_" for c in os.path.basename(file_path)])
                    filename = f"{clean_name}_p{i+1}_box{j}_auto.jpg"
                    
                    # 10% of data goes to the holdout set for manual verification
                    is_holdout = random.random() < 0.10
                    target_dir = holdout_img_dir if is_holdout else img_dir
                    target_csv = holdout_csv_path if is_holdout else csv_path
                    
                    filepath = os.path.join(target_dir, filename)
                    cv2.imwrite(filepath, box.crop_img)
                    
                    # Safely write the CSV row using python's built-in writer
                    with open(target_csv, "a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([filename, text_guess])
                    
    doc.close()

if __name__ == "__main__":
    drawings_dir = Path(r"D:\ai model\drawings")
    
    # Recursively find all PDF and TIF files
    all_files = []
    for ext in ['*.pdf', '*.tif', '*.tiff']:
        all_files.extend(list(drawings_dir.rglob(ext)))
        
    print(f"Found {len(all_files)} total engineering drawings to process!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading our newly trained custom TrOCR model for Self-Training...")
    
    model_path = r"D:\eOCR\trained_model"
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    
    for idx, file_path in enumerate(all_files):
        print(f"\n--- Processing File {idx+1}/{len(all_files)} ---")
        try:
            auto_label_dataset(str(file_path), processor, model, device, output_dir=r"D:\eOCR\full_training_dataset")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            
    print("\nSelf-Training dataset generation complete! You can now run train_trocr.py again to absorb this knowledge!")
