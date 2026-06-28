import cv2
import fitz
import numpy as np
import os

# eDOCr imports
from eDOCr import keras_ocr
from eDOCr import tools
import string

def test_keras_ocr_on_crops(pdf_path):
    print("Loading eDOCr's specialized keras-ocr model...")
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
    
    print("Extracting boxes from page 1...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
    class_list, img_boxes = tools.box_tree.findrect(img)
    boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
    
    raw_lists = [class_list, gdt_boxes, boxes_infoblock]
    all_boxes = []
    for lst in raw_lists:
        if isinstance(lst, list):
            for item in lst:
                if hasattr(item, 'crop_img'): all_boxes.append(item)
                elif isinstance(item, list):
                    for sub_item in item:
                        if hasattr(sub_item, 'crop_img'): all_boxes.append(sub_item)
        elif hasattr(lst, 'crop_img'): all_boxes.append(lst)
            
    print(f"Found {len(all_boxes)} boxes. Testing the first 10 with keras-ocr...")
    
    count = 0
    for j, box in enumerate(all_boxes):
        if count >= 10: break
        if box.crop_img is None or box.crop_img.shape[0] == 0 or box.crop_img.shape[1] == 0: continue
        height, width = box.crop_img.shape[:2]
        if height > 1000 or width > 2000: continue
        
        try:
            img1 = [box.crop_img]
            preds = pipeline.recognize(img1)[0]
            if len(preds) > 0:
                text_guess = " ".join([p[0] for p in preds])
            else:
                text_guess = ""
        except Exception as e:
            text_guess = ""
            
        print(f"Box {j}: {text_guess}")
        count += 1

if __name__ == "__main__":
    test_keras_ocr_on_crops(r"D:\DRG & Model\phase1\34  006506555B1 Agni Sleeve Clutch Release_Rev. AD.pdf")
