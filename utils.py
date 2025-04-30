import face_recognition
import os
import pandas as pd
from datetime import datetime

def encode_faces():
    known_encodings = []
    known_names = []
    for file in os.listdir("faces"):
        img = face_recognition.load_image_file(f"faces/{file}")
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(file.split('.')[0])
    return known_encodings, known_names

def mark_attendance(name):
    os.makedirs("attendance", exist_ok=True)
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')
    path = f"attendance/{date}.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])

    if name not in df['Name'].values:
        df = df.append({'Name': name, 'Time': time}, ignore_index=True)
        df.to_csv(path, index=False)
        print(f"Attendance marked for {name}")