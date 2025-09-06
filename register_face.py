# register_face.py
import cv2
import os
import time
from utils import ensure_person_dir, train_and_save_recognizer, detect_faces

def register_face(name, samples=30, wait_between=0.1):
    name = name.strip().replace(" ", "_")
    if not name:
        print("Invalid name.")
        return

    person_dir = ensure_person_dir(name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print(f"[INFO] Capturing {samples} samples for {name}. Press 'q' to quit early.")
    count = 0
    while count < samples:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            # resize to consistent size
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(person_dir, f"{name}_{int(time.time()*1000)}_{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/{samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            break  # only save one face per frame

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(int(wait_between*1000)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    success, msg = train_and_save_recognizer()
    if success:
        print("[INFO] Registration complete and model updated.")
    else:
        print("[WARN]", msg)
