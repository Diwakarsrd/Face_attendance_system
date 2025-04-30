import cv2
import os

def register_face(name):
    cam = cv2.VideoCapture(0)
    os.makedirs("faces", exist_ok=True)
    print("Press 's' to capture and save the face.")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            path = f"faces/{name}.jpg"
            cv2.imwrite(path, frame)
            print(f"Face saved as {path}")
            break

    cam.release()
    cv2.destroyAllWindows()