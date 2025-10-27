import os
import glob
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path("../data/raw_images")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lists to hold data
embeddings = []
labels = []
meta_records = []

print("🔍 Extracting face embeddings...")

# Loop through each person's folder
for label_folder in os.listdir(DATA_DIR):
    folder = DATA_DIR / label_folder
    if not folder.is_dir():
        continue  # skip files if any

    print(f"\nProcessing folder: {label_folder}")
    img_paths = glob.glob(str(folder / "*.jpg")) + glob.glob(str(folder / "*.png"))

    for img_path in tqdm(img_paths):
        try:
            img = face_recognition.load_image_file(img_path)
            # Detect faces (you can switch to model='cnn' if you have GPU)
            face_locations = face_recognition.face_locations(img, model='hog')
            if len(face_locations) == 0:
                continue  # skip if no face detected

            # Compute embeddings
            encodings = face_recognition.face_encodings(img, face_locations)
            for enc in encodings:
                embeddings.append(enc)
                labels.append(label_folder)
                meta_records.append({'image_path': img_path, 'label': label_folder})
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

# Convert to NumPy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save
np.save(OUT_DIR / "embeddings.npy", embeddings)
np.save(OUT_DIR / "labels.npy", labels)
pd.DataFrame(meta_records).to_csv(OUT_DIR / "meta.csv", index=False)

print("\n✅ Done!")
print(f"Total faces encoded: {len(embeddings)}")
print(f"Saved to: {OUT_DIR}/embeddings.npy, labels.npy, and meta.csv")
