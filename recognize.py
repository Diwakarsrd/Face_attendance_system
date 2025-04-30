import cv2
import face_recognition
from utils import encode_faces, mark_attendance
from email_alert import send_alert

def recognize_faces():
    known_encodings, known_names = encode_faces()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = small[:, :, ::-1]

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for enc, loc in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_encodings, enc)
            name = "Unknown"
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                mark_attendance(name)
            else:
                send_alert()

            y1, x2, y2, x1 = [v * 4 for v in loc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()