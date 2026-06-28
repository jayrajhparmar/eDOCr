import cv2
import numpy as np
import sys
sys.path.insert(0, r"D:\eOCR\.venv\Lib\site-packages")
from eDOCr import keras_ocr
from pdf2image import convert_from_path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

poppler_path = r"D:\eOCR\poppler\poppler-24.02.0\Library\bin"
pdf_path = r"D:\DRG & Model\phase1\26  Sleeve 000016323P04.pdf"
images = convert_from_path(pdf_path, poppler_path=poppler_path)
img = np.array(images[0])

# Just test CRAFT on the raw image
print("Loading detector...")
detector = keras_ocr.detection.Detector()
print("Detecting...")
box_groups = detector.detect([img])

overlay = img.copy()
for box in box_groups[0]:
    pts = np.array(box, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)

cv2.imwrite("C:/Users/jayra/.gemini/antigravity/brain/0b1a7102-abfd-4d8d-b2f3-fcc4c48db9b5/raw_craft_boxes.jpg", overlay)
print(f"Total boxes found by CRAFT: {len(box_groups[0])}")
