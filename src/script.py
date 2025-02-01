import os
import shutil
from sklearn.model_selection import train_test_split

# Original images
image_folder = "../data/synthetic/fish_images"
output_folder = "../data/yolo_dataset"

# Create directories if missing
os.makedirs(f"{output_folder}/images/train", exist_ok=True)
os.makedirs(f"{output_folder}/images/val", exist_ok=True)

# Split images (80% train, 20% validation)
images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

# Copy images to YOLO folders
for img in train_imgs:
    shutil.copy(f"{image_folder}/{img}", f"{output_folder}/images/train/{img}")
for img in val_imgs:
    shutil.copy(f"{image_folder}/{img}", f"{output_folder}/images/val/{img}")