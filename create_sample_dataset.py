import os
import cv2
import numpy as np

# Create directories
base_dir = r"D:\eOCR\sample_dataset"
img_dir = os.path.join(base_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# Function to create a mock image with text
def create_mock_image(filename, text_to_draw):
    # Create a blank white image (height: 50, width: 150, 3 channels)
    img = np.ones((50, 150, 3), dtype=np.uint8) * 255
    
    # Add the text to the image in black
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text_to_draw, (10, 35), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Save the image
    cv2.imwrite(os.path.join(img_dir, filename), img)

# Data for our sample dataset
samples = [
    ("img_001.jpg", "42.0"),
    ("img_002.jpg", "R5"),
    ("img_003.jpg", "Concentricity 0.1")
]

# Generate the images
for filename, text in samples:
    create_mock_image(filename, text)

# Generate the labels.csv file
csv_path = os.path.join(base_dir, "labels.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("filename,text\n")
    for filename, text in samples:
        f.write(f"{filename},{text}\n")

print(f"Successfully created sample dataset at: {base_dir}")
