# recognize.py
import cv2
import time
import threading
from utils import load_recognizer, detect_faces, mark_attendance
from email_alert import send_alert

UNKNOWN_EMAIL_COOLDOWN = 60  # seconds between alert emails for unknown faces

_last_unknown_alert = 0

def _send_alert_thread():
    try:
        send_alert()
    except Exception as e:
        print("Alert send failed:", e)

def recognize_faces(confidence_threshold=70):
    """
    Runs webcam, recognizes faces using LBPH recognizer.
    confidence_threshold: lower means stricter match (LBPH returns smaller = better).
    """
    recognizer, label_map = load_recognizer()
    if recognizer is None:
        print("[WARN] No trained model found. Register faces first (option 1).")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("[INFO] Starting recognition. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_img, (200, 200))
            except Exception:
                continue

            label, conf = recognizer.predict(face_resized)  # conf: lower = better
            name = "Unknown"
            if conf <= confidence_threshold:
                name = label_map.get(label, "Unknown")
                mark_attendance(name)
                color = (0, 200, 0)
                text = f"{name} ({conf:.1f})"
            else:
                color = (0, 0, 255)
                text = f"Unknown ({conf:.1f})"
                # send email alert but restrict to cooldown
                global _last_unknown_alert
                now = time.time()
                if now - _last_unknown_alert > UNKNOWN_EMAIL_COOLDOWN:
                    _last_unknown_alert = now
                    threading.Thread(target=_send_alert_thread, daemon=True).start()

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance - Press ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
