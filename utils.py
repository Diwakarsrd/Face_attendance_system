# utils.py
import os
import cv2
import pandas as pd
import datetime
import numpy as np

FACES_DIR = "faces"
MODEL_PATH = "models/lbph_model.yml"
LABELS_PATH = "models/labels.npy"
ATTENDANCE_DIR = "attendance"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def ensure_person_dir(name):
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir

def get_images_and_labels():
    """
    Read images from faces/<name>/ and return face samples and numeric labels plus mapping.
    """
    label_map = {}
    faces = []
    labels = []
    current_label = 0

    for name in sorted(os.listdir(FACES_DIR)):
        person_dir = os.path.join(FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        label_map[current_label] = name
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        return [], [], {}
    return faces, np.array(labels), label_map

def train_and_save_recognizer():
    """
    Train LBPH recognizer from faces/ and save model + labels mapping.
    """
    faces, labels, label_map = get_images_and_labels()
    if len(faces) == 0:
        return False, "No faces found to train."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(MODEL_PATH)
    # save mapping (index -> name)
    np.save(LABELS_PATH, label_map)
    return True, "Model trained and saved."

def load_recognizer():
    """
    Load recognizer and labels mapping. Returns (recognizer, label_map) or (None, {}) if missing.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()
    return recognizer, label_map

def mark_attendance(name, extra=None):
    """
    Append attendance row to CSV for today's date (one CSV per day).
    """
    date_str = datetime.date.today().isoformat()
    fname = os.path.join(ATTENDANCE_DIR, f"attendance_{date_str}.csv")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"name": name, "timestamp": now}
    if extra:
        row.update(extra)
    df = pd.DataFrame([row])
    if not os.path.exists(fname):
        df.to_csv(fname, index=False)
    else:
        df.to_csv(fname, mode="a", header=False, index=False)

def detect_faces(gray_frame):
    """
    Return list of bounding boxes (x, y, w, h) using Haar cascade.
    """
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces
