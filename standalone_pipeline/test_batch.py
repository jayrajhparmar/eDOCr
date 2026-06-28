import os
import sys
import json
import glob
from pathlib import Path

# Ensure we can import the app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_runtime import LocalOCRRuntime

def test_batch_drawings():
    pdf_dir = r"D:\JAYRAJ\01-HENNA\BHT\ding dong\Fig 602\Machining\2''\PDF"
    results_dir = r"D:\eOCR\standalone_pipeline\batch_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    pdf_files = [os.path.join(pdf_dir, "M-0602FB0200-080.pdf")]
    
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return
        
    print(f"Found {len(pdf_files)} drawings! Starting the batch testing pipeline...")
    print("-" * 50)
    
    # Initialize the OCR Runtime (loads AI into GPU once)
    ocr = LocalOCRRuntime()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"\n[{i}/{len(pdf_files)}] Processing: {filename}...")
        
        try:
            result = ocr.extract_pdf_text(pdf_path)
            
            output_file = os.path.join(results_dir, f"{os.path.splitext(filename)[0]}_output.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"--- RAW TEXT FROM {filename} ---\n\n")
                f.write(result["text"])
                f.write("\n\n--- STRUCTURED JSON (if LLM is running) ---\n\n")
                f.write(json.dumps(result.get("structured_data"), indent=2))
                
            if "page_images" in result and result["page_images"]:
                image_file = os.path.join(results_dir, f"{os.path.splitext(filename)[0]}_labeled.png")
                result["page_images"][0].save(image_file)
                
            print(f"  -> Success! Results saved to batch_results/{os.path.basename(output_file)}")
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")

if __name__ == "__main__":
    test_batch_drawings()
