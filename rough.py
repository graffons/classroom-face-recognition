from PIL import Image
import glob
import os

base_path = "data/raw_images"

for person_folder in os.listdir(base_path):
    folder = os.path.join(base_path, person_folder)
    if not os.path.isdir(folder):
        continue

    for img_path in glob.glob(os.path.join(folder, "*.jfif")) + glob.glob(os.path.join(folder, "*.jpeg")):
        try:
            img = Image.open(img_path)
            new_path = os.path.splitext(img_path)[0] + ".jpg"  # same name, .jpg extension
            img.save(new_path, "JPEG")
            print(f"✅ Converted: {img_path} → {new_path}")
        except Exception as e:
            print(f"⚠️ Failed to convert {img_path}: {e}")













