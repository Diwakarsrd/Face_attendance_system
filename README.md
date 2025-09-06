# Face Recognition Attendance System

This is a Python-based face recognition attendance system using webcam. It allows you to register faces, recognize them in real time, and log attendance.

## Features
- Face registration via webcam
- Real-time face recognition
- Attendance logging to CSV (per day)
- Email alert on unknown face detection

## Folder Structure
- `faces/` – stores known face images
- `attendance/` – stores daily logs
- `main.py` – app launcher
- `register_face.py` – register new faces
- `recognize.py` – run real-time recognition

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Email Alerts
Edit `email_alert.py` to configure your email credentials.

> Make sure to allow "less secure apps" or use an app password for Gmail.

## Author
Diwakar S