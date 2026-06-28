import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def test_model(image_path, model_dir):
    print(f"Loading custom model from {model_dir}...")
    
    # Load the fine-tuned model and processor
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate prediction
    print("Generating prediction...")
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n===============================")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Predicted Text: {generated_text}")
    print("===============================\n")

if __name__ == "__main__":
    # Test on a known crop from the eval set
    # Make sure to replace this with a valid filename if this one doesn't exist
    eval_dir = r"D:\eOCR\cleaned_dataset\eval\images"
    
    if os.path.exists(eval_dir):
        sample_images = os.listdir(eval_dir)
        if len(sample_images) > 0:
            test_image = os.path.join(eval_dir, sample_images[0])
            model_path = r"D:\eOCR\custom_trocr_model\final_model"
            
            if os.path.exists(model_path):
                test_model(test_image, model_path)
            else:
                print(f"Model not found at {model_path}. Please train it first!")
        else:
            print("No images found in eval dataset!")
    else:
        print("Eval dataset directory not found!")
