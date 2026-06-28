import cv2
import fitz
import numpy as np
from eDOCr import tools
from paddleocr import PaddleOCR
import os

def test_paddle_on_crops(pdf_path):
    print("Testing PaddleOCR on the crops extracted by eDOCr...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
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
            
    print(f"Found {len(all_boxes)} boxes. Testing the first 10...")
    
    count = 0
    for j, box in enumerate(all_boxes):
        if count >= 10: break
        if box.crop_img is None or box.crop_img.shape[0] == 0 or box.crop_img.shape[1] == 0: continue
        height, width = box.crop_img.shape[:2]
        if height > 1000 or width > 2000: continue
        
        result = ocr.ocr(box.crop_img, cls=True)
        text_guess = ""
        if result and result[0]:
            text_guess = " ".join([line[1][0] for line in result[0]])
            
        print(f"Box {j}: {text_guess}")
        count += 1

if __name__ == "__main__":
    test_paddle_on_crops(r"D:\DRG & Model\phase1\34  006506555B1 Agni Sleeve Clutch Release_Rev. AD.pdf")
