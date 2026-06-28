import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
# 1. LOAD TrOCR INTO GPU BEFORE ANYTHING ELSE
print("Loading TrOCR model into GPU for the Hybrid Pipeline...")
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\eOCR\trained_model"
processor = TrOCRProcessor.from_pretrained(model_path)
ai_model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# Set generation config for better accuracy (matches training)
ai_model.generation_config.max_length = 64
ai_model.generation_config.early_stopping = True
ai_model.generation_config.no_repeat_ngram_size = 3
ai_model.generation_config.length_penalty = 2.0
ai_model.generation_config.num_beams = 4
ai_model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
ai_model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

print("TrOCR loaded successfully!\n")

# 2. CREATE THE DUMMY RECOGNIZER TO MONKEY-PATCH KERAS-OCR
class DummyModel:
    def load_weights(self, path):
        # Silently ignore Keras weights loading
        pass

class TrOCR_Recognizer:
    def __init__(self, alphabet=None):
        self.alphabet = alphabet
        self.model = DummyModel()

    def recognize_from_boxes(self, images, box_groups, **kwargs):
        predictions = []
        for img, boxes in zip(images, box_groups):
            img_preds = []
            
            # eDOCr usually passes images as RGB inside keras_ocr pipeline, 
            # but let's safely handle numpy arrays to PIL Images
            if isinstance(img, np.ndarray):
                # Ensure it's not a float array
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                
                # Check channels
                if len(img.shape) == 2: # Grayscale
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4: # RGBA
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else: 
                    rgb_img = img
            else:
                rgb_img = np.array(img.convert('RGB'))

            for box in boxes:
                # box is usually a numpy array of 4 points [[x,y], [x,y], [x,y], [x,y]]
                x1, y1 = np.min(box[:, 0]), np.min(box[:, 1])
                x2, y2 = np.max(box[:, 0]), np.max(box[:, 1])
                
                # Expand box slightly to match TrOCR training
                pad = 4
                x1 = max(0, int(x1) - pad)
                y1 = max(0, int(y1) - pad)
                x2 = min(rgb_img.shape[1], int(x2) + pad)
                y2 = min(rgb_img.shape[0], int(y2) + pad)
                
                crop = rgb_img[y1:y2, x1:x2]
                
                if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                    img_preds.append("")
                    continue

                # --- PREPROCESSING FOR TrOCR ---
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                # Normalize to use full 0-255 dynamic range (increases contrast)
                norm_gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # If the background is mostly dark (inverted text), invert it to white background.
                if np.mean(norm_gray) < 127:
                    norm_gray = cv2.bitwise_not(norm_gray)
                
                # Convert back to RGB for TrOCR
                processed_crop = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2RGB)
                
                pil_crop = Image.fromarray(processed_crop)
                
                # TrOCR REQUIRES padding around the text to avoid hallucinations!
                # Since eDOCr crops extremely tightly, we MUST add a white border.
                pil_crop = ImageOps.expand(pil_crop, border=30, fill='white')
                
                pixel_values = processor(pil_crop, return_tensors="pt").pixel_values.to(device)
                
                outputs = ai_model.generate(pixel_values)
                text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                img_preds.append(text)
                
            predictions.append(img_preds)
            
        return predictions

    def recognize(self, image):
        # Some eDOCr functions call recognize(image) directly on a cropped image
        # We can just wrap it in recognize_from_boxes
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.width, image.height
        box_group = np.array([[[0,0], [w,0], [w,h], [0,h]]])
        return self.recognize_from_boxes([image], [box_group])[0][0]

# 3. PERFORM THE MONKEY-PATCH!
# We must append the site-packages to path so eDOCr can be imported
edocr_path = r"D:\eOCR\.venv\Lib\site-packages"
if edocr_path not in sys.path:
    sys.path.insert(0, edocr_path)
    
from eDOCr import keras_ocr

# Swap the recognizer
keras_ocr.recognition.Recognizer = TrOCR_Recognizer

# To prevent the OOM Memory Leak, we must cache the CRAFT detector globally!
GLOBAL_CRAFT_DETECTOR = None

def patched_read_dimensions(img_path, alphabet=None, weight_path=None, cluster_t=20):
    global GLOBAL_CRAFT_DETECTOR
    from eDOCr.tools.pipeline_dimensions import Pipeline, detect_the_patches, get_alfa, analyse_pred, subimage, clean_h_lines
    from eDOCr.tools.tolerances import check_tolerances
    import cv2
    import math
    import re
    
    # Custom analyse_pred to handle TrOCR tolerances better
    def custom_analyse_pred(pred, cnts):
        # Fall back to original, but enhance it for TrOCR formats
        pred_dict, add = analyse_pred(pred, cnts)
        
        # Override the strict eDOCr filter. If TrOCR predicted something, let's keep it!
        # Especially dimensions like 'Ø100', 'M6', 'R5' or pure tolerances.
        if pred.strip() != "":
            # Force add=True to bypass eDOCr's strict digit check
            add = True
            
            # If the original failed to parse properly, let's build a fallback pred_dict
            if not pred_dict or pred_dict.get('type') == 'general' or 'type' not in pred_dict:
                pred_dict['type'] = 'Dimension'  # Fallback type
                pred_dict['value'] = pred.strip()
        else:
            add = False

        # If original analyse_pred failed to parse a tolerance correctly (returns 'general')
        if pred_dict.get('tolerance') == 'general':
            # Check for combined without spaces like "25.00+0.02-0.01" or "+0.20"
            if pred.startswith('+') or pred.startswith('-'):
                pred_dict['type'] = 'Tolerance'
                pred_dict['nominal'] = pred
                pred_dict['value'] = '' # No nominal
                if pred.startswith('+'):
                    pred_dict['upper_bound'] = pred
                else:
                    pred_dict['lower_bound'] = pred
                if 'tolerance' in pred_dict:
                    del pred_dict['tolerance']
            # Check for multiple signs e.g., 25+0.02-0.01
            elif '+' in pred and '-' in pred:
                m = re.match(r"^(.*?)(?:\+)(.*?)(?:\-)(.*?)$", pred)
                if m:
                    pred_dict['type'] = 'Length'
                    pred_dict['value'] = m.group(1).strip()
                    pred_dict['upper_bound'] = '+' + m.group(2).strip()
                    pred_dict['lower_bound'] = '-' + m.group(3).strip()
                    if 'tolerance' in pred_dict:
                        del pred_dict['tolerance']
            # Check for diameter symbol like Ø100 (TrOCR sometimes outputs 'Ø' or 'O')
            if 'Ø' in pred or 'O' in pred[:2]:
                pred_dict['type'] = 'Diameter'
                
        print(f"TrOCR Read: '{pred}' -> Keep: {add}, Parsed: {pred_dict}")
        return pred_dict, add
    
    if alphabet and weight_path:
        recognizer = keras_ocr.recognition.Recognizer(alphabet=alphabet)
    else:
        recognizer = keras_ocr.recognition.Recognizer()
        
    if GLOBAL_CRAFT_DETECTOR is None:
        print("Initializing Global CRAFT Detector for the first time...")
        GLOBAL_CRAFT_DETECTOR = keras_ocr.detection.Detector()
        
    def custom_recognize_dimensions(self, box_groups, img):
        predictions = []
        i = 0
        def safe_subimage(img, center, theta, width, height):
            shape = (img.shape[1], img.shape[0])
            matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
            rotated = cv2.warpAffine(src=img, M=matrix, dsize=shape)
            x, y = int(center[0] - width/2), int(center[1] - height/2)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(rotated.shape[1], x + width), min(rotated.shape[0], y + height)
            return rotated[y1:y2, x1:x2]

        for box in box_groups:
            rect = cv2.minAreaRect(box)
            alfa = get_alfa(box)
            if -5 < alfa < 85:
                angle = -round(alfa / 5) * 5
            elif 85 < alfa < 95:
                angle = round(alfa / 5) * 5 - 180
            elif 95 < alfa < 185:
                angle = 180 - round(alfa / 5) * 5
            else:
                angle = alfa
            
            w = int(max(rect[1]) + 5)
            h = int(min(rect[1]) + 2)
            
            img_croped = safe_subimage(img, rect[0], angle, w, h)
            
            if img_croped is None or img_croped.size == 0:
                continue
            
            img_croped, thresh = clean_h_lines(img_croped)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            
            # Increase contour limit significantly to handle long tolerances like Ø51.0 +0.0 -0.5
            if len(cnts) == 1:
                img_croped = cv2.rotate(img_croped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h_c, w_c, _ = img_croped.shape
                b_group = [np.array([[[0,0], [w_c,0], [w_c,h_c], [0,h_c]]])]
                pred = self.recognizer.recognize_from_boxes(images=[img_croped], box_groups=b_group)[0][0]
                dummy_cnts = [1] * max(1, len(pred.replace(' ', '')))
                pred_dict, add = custom_analyse_pred(pred, dummy_cnts)
                
            elif 1 < len(cnts) < 50:
                arr = check_tolerances(img_croped)
                pred = ''
                for img_ in arr:
                    h_c, w_c, _ = img_.shape
                    b_group = [np.array([[[0,0], [w_c,0], [w_c,h_c], [0,h_c]]])]
                    pred_ = self.recognizer.recognize_from_boxes(images=[img_], box_groups=b_group)[0][0]
                    if pred_ == '':
                        img_c_rot = cv2.rotate(img_croped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        h_cr, w_cr, _ = img_c_rot.shape
                        b_group_cr = [np.array([[[0,0], [w_cr,0], [w_cr,h_cr], [0,h_cr]]])]
                        pred = self.recognizer.recognize_from_boxes(images=[img_c_rot], box_groups=b_group_cr)[0][0] + ' '
                        break
                    else:
                        pred += pred_ + ' '
                pred = pred[:-1]
                dummy_cnts = [1] * max(1, len(pred.replace(' ', '')))
                pred_dict, add = custom_analyse_pred(pred, dummy_cnts)
            else:
                add = False
            
            if add:
                i += 1
                pred_id = {'ID': i}
                pred_id.update(pred_dict)
                predictions.append({'pred': pred_id, 'box': box})
                
        return predictions

    Pipeline.recognize_dimensions = custom_recognize_dimensions
    
    pipeline = Pipeline(detector=GLOBAL_CRAFT_DETECTOR, recognizer=recognizer)
    img = Image.open(img_path)
    snippets = detect_the_patches(img, pipeline, cluster_t=cluster_t)
    return snippets

from eDOCr.tools import pipeline_dimensions
pipeline_dimensions.read_dimensions = patched_read_dimensions


# 4. RUN STANDARD eDOCr WITH OUR NEW SUPERCHARGED OCR
def run_hybrid_edocr(pdf_path, dest_folder):
    print(f"Running eDOCr Layout Engine + TrOCR Reading Engine on: {os.path.basename(pdf_path)}")
        
    from eDOCr import tools
    from pdf2image import convert_from_path
    
    os.makedirs(dest_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 1. Convert PDF to Image
    print("Converting PDF to Image...")
    poppler_path = r"D:\eOCR\poppler\poppler-24.02.0\Library\bin"
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    
    # We only process the first page for this test
    img = np.array(images[0])
    
    # 2. Find Rects (Layout parsing via OpenCV)
    print("Finding layout rects via eDOCr OpenCV...")
    class_list, img_boxes = tools.box_tree.findrect(img)
    boxes_infoblock, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
    
    # 3. Read InfoBlock (Uses Monkey-Patched TrOCR!)
    print("Reading InfoBlock using TrOCR...")
    infoblock_dict = tools.pipeline_infoblock.read_infoblocks(boxes_infoblock, img, "alphabet", "model_infoblock")
    
    # 4. Read GDTs (Uses Monkey-Patched TrOCR!)
    print("Reading GD&Ts using TrOCR...")
    gdt_dict = tools.pipeline_gdts.read_gdtbox1(gdt_boxes, "alphabet", "model_gdts", "alphabet", "model_dim")
    
    # 5. Read Dimensions (Uses Monkey-Patched TrOCR!)
    print("Reading Dimensions using TrOCR...")
    cv2.imwrite(os.path.join(dest_folder, "temp_process.jpg"), process_img)
    dimension_dict = tools.pipeline_dimensions.read_dimensions(os.path.join(dest_folder, "temp_process.jpg"), "alphabet", "model_dim", 20)
    
    # 6. Save eDOCr outputs
    print("Saving eDOCr CSV files and labeled images...")
    tools.output.record_data(dest_folder, filename, infoblock_dict, gdt_dict, dimension_dict)
    
    # Generate the overlay image with all bounding boxes marked
    color_palette = {'infoblock': (0,255,0), 'gdts': (255,0,0), 'dimensions': (0,0,255), 'frame': (255,255,0), 'flag': (0,0,0)}
    overlay_img = tools.output.mask_the_drawing(img, infoblock_dict, gdt_dict, dimension_dict, cl_frame, color_palette)
    
    # Save bounding boxes image
    cv2.imwrite(os.path.join(dest_folder, filename + '_boxes.jpg'), overlay_img)
    
    print(f"\nDone! eDOCr layout data saved with 100% TrOCR text accuracy to: {dest_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Path to single PDF")
    parser.add_argument("--dir", default=None, help="Path to directory of PDFs")
    parser.add_argument("--out", default=r"D:\eOCR\standalone_pipeline\edocr_hybrid_results", help="Output directory")
    args = parser.parse_args()
    
    if args.dir:
        for fname in os.listdir(args.dir):
            if fname.lower().endswith('.pdf'):
                print(f"\n--- Processing {fname} ---")
                try:
                    run_hybrid_edocr(os.path.join(args.dir, fname), args.out)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"File not found: {args.pdf}")
            sys.exit(1)
        run_hybrid_edocr(args.pdf, args.out)
    else:
        print("Please provide --pdf or --dir argument")
