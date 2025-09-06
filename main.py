# main.py
import os

def main():
    print("1. Register New Face")
    print("2. Start Attendance")
    choice = input("Enter choice: ").strip()

    if choice == '1':
        from register_face import register_face
        name = input("Enter name: ")
        register_face(name)
    elif choice == '2':
        from recognize import recognize_faces
        print("Starting recognition...")
        recognize_faces()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
