import os
# Set huggingface cache AND temp directories to D drive since C drive is completely full
os.environ["HF_HOME"] = r"D:\eOCR\.cache"
os.environ["TMPDIR"] = r"D:\eOCR\tmp"
os.environ["TEMP"] = r"D:\eOCR\tmp"
os.environ["TMP"] = r"D:\eOCR\tmp"
os.makedirs(r"D:\eOCR\tmp", exist_ok=True)

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from eDOCr import tools
import torch

def auto_label_dataset(pdf_path, processor, model, device, output_dir="full_training_dataset"):
    print(f"\nGenerating dataset from {os.path.basename(pdf_path)}...")
    
    # Setup directories
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "labels.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filename,text\n")
            
    # 1. Convert PDF to image at 300 DPI
    print("Converting PDF to images (this might take a moment)...")
    doc = fitz.open(pdf_path)
    
    for i in range(len(doc)):
        print(f"Processing page {i+1}...")
        page = doc[i]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Convert RGB to cv2 BGR
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # 2. Extract bounding boxes using OpenCV
        print("Extracting bounding boxes using eDOCr computer vision logic...")
        class_list, img_boxes = tools.box_tree.findrect(img)
        boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
        
        # Combine all boxes
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
        print("Passing crops to Microsoft TrOCR for labeling...")
        
        with open(csv_path, "a", encoding="utf-8") as f:
            for j, box in enumerate(all_boxes):
                if box.crop_img is None or box.crop_img.shape[0] == 0 or box.crop_img.shape[1] == 0:
                    continue
                
                # Check for enormous garbage crops (like page borders)
                height, width = box.crop_img.shape[:2]
                if height > 1000 or width > 2000:
                    continue # Skip massive page borders
                
                # TrOCR expects RGB PIL images
                img_rgb = cv2.cvtColor(box.crop_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # 3. Predict the text
                try:
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                    # Suppress generate config warnings by specifying max_new_tokens
                    generated_ids = model.generate(pixel_values, max_new_tokens=30)
                    text_guess = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                except Exception as e:
                    print(f"Prediction failed for box {j}: {e}")
                    text_guess = ""
                
                # Only save if it actually found text
                text_guess = text_guess.strip()
                if text_guess != "":
                    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                    clean_name = "".join([c if c.isalnum() else "_" for c in pdf_name])
                    filename = f"{clean_name}_p{i+1}_box{j}_auto.jpg"
                    filepath = os.path.join(img_dir, filename)
                    cv2.imwrite(filepath, box.crop_img)
                    
                    # Clean up guess for CSV
                    text_guess = text_guess.replace(",", "") # remove commas so it doesn't break CSV
                    text_guess = text_guess.replace("\n", " ") # remove newlines
                    f.write(f"{filename},{text_guess}\n")
                    
                    print(f"Labeled {filename}: {text_guess}")
                    
    print(f"Done! Labeled dataset saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import glob
    drawings_dir = r"D:\ai model\drawings"
    pdfs = glob.glob(os.path.join(drawings_dir, "*.pdf"))
    print(f"Found {len(pdfs)} PDF drawings to process.")
    
    # Load model ONCE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading Microsoft TrOCR model from local folder...")
    processor = TrOCRProcessor.from_pretrained(r"D:\eOCR\trocr-small")
    model = VisionEncoderDecoderModel.from_pretrained(r"D:\eOCR\trocr-small").to(device)
    print("Model ready!")
    
    for pdf_file in pdfs:
        try:
            auto_label_dataset(pdf_file, processor, model, device, output_dir=r"D:\eOCR\full_training_dataset")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
