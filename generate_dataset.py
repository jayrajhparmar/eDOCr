import os
import cv2
import glob
import pandas as pd
from pdf2image import convert_from_path
from eDOCr import keras_ocr
from eDOCr import tools
from anytree import RenderTree, NodeMixin

def generate_dataset(pdf_path, output_dir="training_data"):
    print(f"Generating dataset from {pdf_path}...")
    
    # Setup directories
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "labels.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("filename,text\n")
            
    import string
    Extra='(),.+-±:/°"⌀'
    alphabet_dimensions=string.digits + 'AaBCDRGHhMmnx'+ Extra
    model_dimensions=keras_ocr.tools.download_and_verify(
        url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_dimensions.h5",
        filename="recognizer_dimensions.h5",
        sha256="a1c27296b1757234a90780ccc831762638b9e66faf69171f5520817130e05b8f",
    )
    recognizer = keras_ocr.recognition.Recognizer(alphabet=alphabet_dimensions)
    recognizer.model.load_weights(model_dimensions)
    pipeline = keras_ocr.pipeline.Pipeline(recognizer=recognizer, scale=2)
    
    # 1. Convert PDF to image at 300 DPI (sweet spot)
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=300)
    
    for i, pil_img in enumerate(images):
        print(f"Processing page {i+1}...")
        
        # Convert PIL to cv2 BGR
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # 2. Extract bounding boxes using OpenCV
        print("Extracting bounding boxes...")
        class_list, img_boxes = tools.box_tree.findrect(img)
        boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
        
        # Combine all boxes we care about (dimensions and GD&Ts)
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
        
        print(f"Found {len(all_boxes)} text regions. Processing and saving...")
        
        with open(csv_path, "a", encoding="utf-8") as f:
            for j, box in enumerate(all_boxes):
                if box.crop_img is None or box.crop_img.shape[0] == 0 or box.crop_img.shape[1] == 0:
                    continue
                
                # We need to run the recognizer to get the guess.
                # eDOCr wraps it like this:
                img1 = [box.crop_img]
                try:
                    preds = pipeline.recognize(img1)[0]
                    # preds is a list of (text, box)
                    if len(preds) > 0:
                        # Combine text if there are multiple detected words in the crop
                        text_guess = " ".join([p[0] for p in preds])
                    else:
                        text_guess = ""
                except Exception as e:
                    text_guess = ""
                
                # Only save if it actually found text
                if text_guess != "":
                    filename = f"page{i+1}_box{j}.jpg"
                    filepath = os.path.join(img_dir, filename)
                    cv2.imwrite(filepath, box.crop_img)
                    
                    # Clean up guess for CSV
                    text_guess = text_guess.replace(",", "") # remove commas so it doesn't break CSV
                    f.write(f"{filename},{text_guess}\n")
                    
    print(f"Done! Dataset saved to {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import numpy as np # import here to avoid global namespace issues if not needed
    
    pdf = r"D:\DRG & Model\phase1\34  006506555B1 Agni Sleeve Clutch Release_Rev. AD.pdf"
    generate_dataset(pdf, output_dir=r"D:\eOCR\full_training_dataset")
