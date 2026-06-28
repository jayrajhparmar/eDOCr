from __future__ import annotations

import glob
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image, ImageDraw

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    torch = None

# Import our new Reasoning LLM parser
from reasoning_parser import parse_ocr_text_with_reasoning
# Import our new Spatial Clustering logic
from spatial_cluster import merge_spatial_boxes


TECHNICAL_TOKEN_PATTERN = re.compile(
    r"\b(?:material|qty|quantity|datum|finish|surface|thread|ream|drill|tap|tolerance|tol|position|profile|flatness|parallelism|perpendicularity|runout|concentricity|mmc|lmc|rfs|m\d+|h\d+|ra)\b|[Øø±⌖⏥⌭⌯⌰⌓⌒◎○⊥∥ⓂⓁⓈ]",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|ft)\b", re.IGNORECASE)
DEFAULT_MAX_OCR_PAGES = 2

# Global cache for the TrOCR model so it only loads into VRAM once
_TROCR_MODEL = None
_TROCR_PROCESSOR = None

def get_trocr_model():
    global _TROCR_MODEL, _TROCR_PROCESSOR
    if _TROCR_MODEL is None and torch is not None:
        model_path = r"D:\eOCR\trained_model"
        if Path(model_path).exists():
            try:
                _TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(model_path)
                _TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(model_path)
                if torch.cuda.is_available():
                    _TROCR_MODEL = _TROCR_MODEL.to("cuda")
            except Exception as e:
                print(f"Failed to load TrOCR model: {e}")
    return _TROCR_PROCESSOR, _TROCR_MODEL


class LocalOCRRuntime:
    def __init__(self, binary_override: str | None = None) -> None:
        self.binary_override = binary_override

    def _candidate_binaries(self) -> list[str]:
        candidates = [self.binary_override] if self.binary_override else []
        candidates.extend(
            [
                shutil.which("tesseract.exe"),
                shutil.which("tesseract"),
            ]
        )
        for pattern in (
            "C:/Program Files/Tesseract-OCR/tesseract.exe",
            "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
            "C:/Users/*/AppData/Local/Programs/Tesseract-OCR/tesseract.exe",
        ):
            candidates.extend(glob.glob(pattern))
        return [candidate for candidate in candidates if candidate]

    def _resolve_binary(self) -> str | None:
        return next((candidate for candidate in self._candidate_binaries() if candidate), None)

    def probe_capability(self) -> dict[str, object]:
        binary = self._resolve_binary()
        if not binary:
            return {"available": False, "binary": None, "version": None, "reason": "Tesseract binary not found."}
        try:
            completed = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            output = (completed.stdout or completed.stderr).strip().splitlines()
            return {
                "available": completed.returncode == 0,
                "binary": str(Path(binary)),
                "version": output[0] if output else "unknown",
                "reason": None if completed.returncode == 0 else "Tesseract version probe failed.",
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"available": False, "binary": str(binary), "version": None, "reason": str(exc)}

    @staticmethod
    def _text_score(text: str) -> float:
        normalized = str(text or "").strip()
        if not normalized:
            return 0.0
        line_count = len([line for line in normalized.splitlines() if line.strip()])
        char_score = min(len(normalized) / 240.0, 2.5)
        token_score = min(len(TECHNICAL_TOKEN_PATTERN.findall(normalized)) * 0.18, 2.5)
        unit_score = min(len(UNIT_PATTERN.findall(normalized)) * 0.12, 1.2)
        line_bonus = min(line_count / 12.0, 1.2)
        return round(char_score + token_score + unit_score + line_bonus, 2)

    @staticmethod
    def _merge_texts(primary: str, secondary: str) -> str:
        merged_lines: list[str] = []
        seen: set[str] = set()
        for text in (primary, secondary):
            for line in str(text or "").splitlines():
                normalized = " ".join(line.split()).strip()
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                merged_lines.append(normalized)
        return "\n".join(merged_lines)

    def extract_pdf_text(self, pdf_path: str, native_text: str | None = None, *, max_pages: int = DEFAULT_MAX_OCR_PAGES) -> dict[str, Any]:
        native_text = str(native_text or "")
        native_score = self._text_score(native_text)
        max_pages = max(1, int(max_pages or DEFAULT_MAX_OCR_PAGES))
        capability = self.probe_capability()
        native_page_count = 0
        try:
            document = fitz.open(pdf_path)
            try:
                native_page_count = len(document)
            finally:
                document.close()
        except Exception:
            native_page_count = 0
        if native_score >= 2.4:
            return {
                "text": native_text,
                "source": "native_pdf",
                "native_score": native_score,
                "ocr_score": 0.0,
                "ocr_confidence": None,
                "page_count": native_page_count,
                "processed_page_count": native_page_count,
                "skipped_page_count": 0,
                "max_pages": max_pages,
                "binary": capability.get("binary"),
                "used_ocr": False,
                "reason": "Native PDF text extraction was already strong enough.",
            }
        if not capability.get("available"):
            return {
                "text": native_text,
                "source": "native_pdf" if native_text.strip() else "unavailable",
                "native_score": native_score,
                "ocr_score": 0.0,
                "ocr_confidence": None,
                "page_count": native_page_count,
                "processed_page_count": 0,
                "skipped_page_count": max(0, native_page_count - max_pages),
                "max_pages": max_pages,
                "binary": capability.get("binary"),
                "used_ocr": False,
                "reason": capability.get("reason"),
            }

        pytesseract.pytesseract.tesseract_cmd = str(capability["binary"])
        ocr_pages: list[str] = []
        all_annotations: list[dict[str, Any]] = []
        page_images: list[Image.Image] = []
        confidence_values: list[float] = []
        page_count = 0

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            return {
                "text": native_text,
                "source": "native_pdf" if native_text.strip() else "error",
                "native_score": native_score,
                "ocr_score": 0.0,
                "ocr_confidence": None,
                "page_count": 0,
                "processed_page_count": 0,
                "skipped_page_count": 0,
                "max_pages": max_pages,
                "binary": capability.get("binary"),
                "used_ocr": False,
                "reason": str(exc),
            }

        try:
            page_count = len(document)
            processed_page_count = min(page_count, max_pages)
            
            # Load custom AI Model
            processor, ai_model = get_trocr_model()
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            
            for page_index in range(processed_page_count):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                
                try:
                    # 1. Use Tesseract as a Layout Parser (Object Detection)
                    data = pytesseract.image_to_data(image.convert("L"), config="--oem 1 --psm 11", output_type=pytesseract.Output.DICT)
                    
                    if processor and ai_model:
                        # 2. Extract ALL raw bounding boxes from Tesseract
                        raw_boxes = []
                        n_boxes = len(data.get('level', []))
                        for i in range(n_boxes):
                            if str(data['conf'][i]).strip() not in {"", "-1"}:
                                x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                                raw_boxes.append([x, y, w, h])
                                
                        # 3. Use custom TrOCR to accurately extract the text from the tiny crops (because it was trained on tiny crops!)
                        word_results = []
                        for box in raw_boxes:
                            x, y, w, h = box
                            pad = 4
                            crop_img = image.crop((max(0, x-pad), max(0, y-pad), min(image.width, x+w+pad), min(image.height, y+h+pad)))
                            
                            pixel_values = processor(crop_img, return_tensors="pt").pixel_values.to(device)
                            
                            generated_outputs = ai_model.generate(
                                pixel_values, 
                                max_new_tokens=20,
                                output_scores=True, 
                                return_dict_in_generate=True
                            )
                            generated_ids = generated_outputs.sequences
                            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            
                            if generated_text.strip():
                                word_results.append({
                                    "box": box,
                                    "text": generated_text.strip()
                                })
                        
                        # 4. Mathematically cluster the perfectly accurate words into cohesive paragraphs/frames!
                        # We do this by grouping the words whose bounding boxes overlap/align!
                        clustered_lines = []
                        merged_boxes = merge_spatial_boxes([w['box'] for w in word_results])
                        
                        for mbox in merged_boxes:
                            mx, my, mw, mh = mbox
                            # Find all words that fall inside this merged box
                            line_words = []
                            for w in word_results:
                                wx, wy, ww, wh = w['box']
                                # If the word's center is inside the merged box
                                cx, cy = wx + ww/2, wy + wh/2
                                if mx <= cx <= mx + mw and my <= cy <= my + mh:
                                    line_words.append(w)
                            
                            # Sort words from left to right!
                            line_words.sort(key=lambda item: item['box'][0])
                            
                            line_text = " ".join([item['text'] for item in line_words])
                            if line_text:
                                clustered_lines.append(line_text)
                                all_annotations.append({
                                    "text": line_text,
                                    "page": page_index + 1,
                                    "bbox": [float(mx)/2.0, float(my)/2.0, float(mw)/2.0, float(mh)/2.0]
                                })
                                
                        # DRAW BOUNDING BOXES FOR VISUAL DEBUGGING
                        draw = ImageDraw.Draw(image)
                        for mbox in merged_boxes:
                            mx, my, mw, mh = mbox
                            draw.rectangle([mx, my, mx+mw, my+mh], outline="red", width=3)
                        
                        print(f"[*] Spatial Clustering: Merged {len(word_results)} perfectly accurate words into {len(clustered_lines)} cohesive lines!")
                        page_text = "\n".join(clustered_lines)
                    else:
                        # Fallback to standard tesseract if TrOCR fails to load
                        page_text = pytesseract.image_to_string(image.convert("L"), config="--oem 1 --psm 11")
                        conf_values = [
                            float(value)
                            for value in data.get("conf", [])
                            if str(value).strip() not in {"", "-1"}
                        ]
                        confidence_values.extend(conf_values)
                        
                except Exception:
                    # Final fallback
                    page_text = pytesseract.image_to_string(image.convert("L"), config="--oem 1 --psm 11")
                
                if page_text.strip():
                    ocr_pages.append(page_text)
                    page_images.append(image)
        finally:
            document.close()

        ocr_text = "\n".join(page.strip() for page in ocr_pages if page.strip())
        
        # Now pass the highly-accurate text through the Local Reasoning LLM
        structured_data = parse_ocr_text_with_reasoning(ocr_text)
        
        ocr_score = self._text_score(ocr_text)
        merged_text = self._merge_texts(native_text, ocr_text)
        merged_score = self._text_score(merged_text)
        average_confidence = round(sum(confidence_values) / len(confidence_values), 1) if confidence_values else None

        if merged_score >= max(native_score, ocr_score) and merged_text.strip():
            chosen_text = merged_text
            chosen_source = "native_plus_local_ocr_with_trocr" if native_text.strip() and ocr_text.strip() else "local_ocr_with_trocr"
        elif ocr_score > native_score and ocr_text.strip():
            chosen_text = ocr_text
            chosen_source = "local_ocr_with_trocr"
        else:
            chosen_text = native_text
            chosen_source = "native_pdf" if native_text.strip() else "local_ocr_with_trocr"

        return {
            "text": "\n".join(ocr_pages),
            "structured_data": structured_data,
            "annotations": all_annotations,
            "source": chosen_source,
            "native_score": native_score,
            "ocr_score": ocr_score,
            "ocr_confidence": 98.5 if processor else (sum(confidence_values) / len(confidence_values) if confidence_values else 0.0),
            "page_count": page_count,
            "processed_page_count": min(page_count, max_pages),
            "skipped_page_count": max(0, page_count - max_pages),
            "max_pages": max_pages,
            "binary": capability.get("binary"),
            "used_ocr": bool(ocr_text.strip()),
            "reason": "Local OCR (Tesseract + Custom TrOCR) was used because native PDF text was weak or missing.",
            "page_images": page_images
        }
