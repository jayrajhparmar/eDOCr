import sys
import os
import traceback

sys.path.insert(0, r"D:\eOCR\.venv\Lib\site-packages")
import cv2
import numpy as np
from PIL import Image

try:
    from eDOCr.keras_ocr import detection
    from eDOCr.tools.pipeline_dimensions import detect_the_patches, Pipeline

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    img_path = r"D:\eOCR\hybrid_test_results\temp_process.jpg"
    img = Image.open(img_path)

    print("Loading detector...")
    detector = detection.Detector()

    class DummyRecognizer:
        def recognize_from_boxes(self, images, box_groups, **kwargs):
            return [["dummy"] for _ in box_groups]
        def recognize(self, image):
            return "dummy"

    pipeline = Pipeline(detector=detector, recognizer=DummyRecognizer())

    # Monkey patch recognize_dimensions to draw boxes and print their sizes
    def fake_recognize(self, box_groups, img_arr):
        out = img_arr.copy()
        print(f"Total boxes found by CRAFT: {len(box_groups)}")
        for box in box_groups:
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(r"D:\eOCR\hybrid_test_results\all_craft_boxes.jpg", out)
        return []

    Pipeline.recognize_dimensions = fake_recognize

    print("Detecting patches...")
    detect_the_patches(img, pipeline, cluster_t=20)
    print("Saved all_craft_boxes.jpg")
except Exception as e:
    print("Error occurred:")
    traceback.print_exc()
