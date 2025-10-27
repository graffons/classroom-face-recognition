import cv2
import numpy as np
import face_recognition
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
from pathlib import Path
import dlib  # direct use for encoding (more stable)

# --- Paths ---
MODEL_PATH = Path("models/best_model_SVM_(linear).joblib")
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)
CSV_FILE = DATA_DIR / "attendance.csv"
DB_FILE = DATA_DIR / "attendance.db"

# --- Load trained classifier ---
print("🔹 Loading trained SVM model...")
clf = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Setup SQLite database ---
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
""")
conn.commit()

# --- Helper functions ---
def mark_attendance(name):
    """Write attendance to CSV and SQLite, avoiding duplicates in the same minute."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(CSV_FILE) if CSV_FILE.exists() else pd.DataFrame(columns=['name', 'timestamp'])
    if not ((df['name'] == name) & (df['timestamp'].str[:16] == timestamp[:16])).any():
        df.loc[len(df)] = [name, timestamp]
        df.to_csv(CSV_FILE, index=False)
        cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
        conn.commit()
        print(f"✅ Attendance marked for {name} at {timestamp}")
    else:
        print(f"⏩ Already marked for {name} recently.")


# --- Load dlib models once (faster, more stable) ---
# --- Load dlib models (paths auto-detected from the face_recognition package) ---
import os
import importlib.util

# Locate the installed face_recognition_models package
models_spec = importlib.util.find_spec("face_recognition_models")
if models_spec is None:
    raise ImportError("⚠️ face_recognition_models package not found. Try: pip install face_recognition_models")

models_dir = os.path.dirname(models_spec.origin)

# Build absolute paths to model files
face_rec_model_path = os.path.join(models_dir, "models", "dlib_face_recognition_resnet_model_v1.dat")
pose_predictor_path = os.path.join(models_dir, "models", "shape_predictor_68_face_landmarks.dat")

# Load dlib models
face_encoder = dlib.face_recognition_model_v1(face_rec_model_path)
shape_predictor = dlib.shape_predictor(pose_predictor_path)


# --- Start webcam feed ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise SystemExit("❌ Could not access webcam. Check permissions or camera availability.")

print("\n🎥 Starting real-time recognition... Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])  # BGR → RGB

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small, model='hog')

    # Encode faces using dlib directly (avoids compute_face_descriptor errors)
    face_encodings = []
    if face_locations:
        dlib_rects = [dlib.rectangle(left, top, right, bottom)
                      for (top, right, bottom, left) in face_locations]
        for rect in dlib_rects:
            try:
                shape = shape_predictor(rgb_small, rect)
                face_descriptor = np.array(face_encoder.compute_face_descriptor(rgb_small, shape))
                face_encodings.append(face_descriptor)
            except Exception as e:
                print(f"⚠️ Skipping face due to encoding error: {e}")

    # Predict each detected face
    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        name = "Unknown"
        confidence = 0.0
        try:
            probs = clf.predict_proba([face_enc])[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            if confidence > 0.5:
                name = clf.classes_[pred_idx]
        except Exception:
            name = clf.predict([face_enc])[0]
            confidence = 1.0

        # Scale back up (since frame was reduced 0.25×)
        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]

        # Draw box + label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Log attendance
        if name != "Unknown":
            mark_attendance(name)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
print("👋 Exited successfully.")

