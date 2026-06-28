import os
import sys
import json
import argparse

# Ensure we can import the app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_runtime import LocalOCRRuntime

def test_full_drawing(pdf_path):
    print(f"Testing the new TrOCR Hybrid Pipeline on: {os.path.basename(pdf_path)}")
    print("This will use Tesseract for layout parsing, our custom TrOCR for text extraction, and spatial clustering to stitch it together!")
    print("-" * 50)
    
    # Initialize the OCR Runtime (this will load the custom TrOCR model into the GPU)
    ocr = LocalOCRRuntime()
    
    # Run the extraction!
    result = ocr.extract_pdf_text(pdf_path)
    
    print("\n[ EXTRACTION RESULTS ]")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)
    print(f"Source used: {result['source']}")
    print(f"Confidence score: {result.get('ocr_confidence', 'N/A')}")
    print(f"Pages processed: {result['processed_page_count']}")
    print("-" * 50)
    print("\n[ STRUCTURED LLM DATA ]")
    print("-" * 50)
    print(json.dumps(result["structured_data"], indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR extraction on a PDF.")
    parser.add_argument("pdf_path", nargs="?", default=r"D:\DRG & Model\08.pdf", help="Path to the PDF file")
    args = parser.parse_args()
    
    target_pdf = args.pdf_path
        
    if not os.path.exists(target_pdf):
        print(f"Error: Could not find file {target_pdf}")
        sys.exit(1)
        
    test_full_drawing(target_pdf)
