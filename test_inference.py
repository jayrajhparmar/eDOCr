import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def test_model():
    model_path = r"D:\eOCR\trained_model"
    print(f"Loading custom model from {model_path}...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # Pick a sample image
    image_dir = r"D:\eOCR\full_training_dataset\images"
    sample_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not sample_images:
        print("No images found for testing.")
        return
        
    import random
    test_images = random.sample(sample_images, min(5, len(sample_images)))
    
    for img_name in test_images:
        test_img_path = os.path.join(image_dir, img_name)
        print(f"\nTesting on image: {test_img_path}")
        image = Image.open(test_img_path).convert("RGB")
        
        pixel_values = processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Extracted Text: '{generated_text}'")
    print("-----------------\n")

if __name__ == "__main__":
    test_model()
