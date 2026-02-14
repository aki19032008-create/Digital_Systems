from django.shortcuts import render
from django.http import StreamingHttpResponse
from .models import Attendance
from django.conf import settings
from datetime import datetime
import cv2
import pandas as pd
import numpy as np
import os
import time

# =====================
# GLOBALS
# =====================
camera = None
attendance_marked = False
TIME_LIMIT = 15  # seconds

# =====================
# LOAD STUDENTS FROM CSV
# =====================
CSV_PATH = os.path.join(settings.BASE_DIR, 'data', 'students.csv')

def load_students():
    df = pd.read_csv(CSV_PATH)

    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for _, row in df.iterrows():
        img_path = os.path.join(settings.BASE_DIR, row['image'])
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            continue

        faces.append(gray)
        labels.append(label_id)

        label_map[label_id] = {
            "roll_no": row['roll_no'],
            "name": row['name'],
            "department": row['department']
        }
        label_id += 1

    return faces, labels, label_map

# =====================
# FACE DETECTOR + RECOGNIZER
# =====================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# =====================
# HOME
# =====================
def home(request):
    return render(request, "app/home.html")

# =====================
# ATTENDANCE PAGE
# =====================
def attendance_register(request):
    records = Attendance.objects.all().order_by('-id')
    return render(request, "app/attendance_register.html", {
        "records": records
    })

# =====================
# STAFF LOGIN
# =====================
def staff_login(request):
    return render(request, "app/staff_login.html")

# =====================
# CAMERA STREAM
# =====================
def gen_frames():
    global camera, attendance_marked

    faces, labels, label_map = load_students()
    recognizer.train(faces, np.array(labels))

    camera = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected:
            roi = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi)

            if confidence < 80 and not attendance_marked:
                student = label_map[label]

                Attendance.objects.create(
                    name=student["name"],
                    date=datetime.now().date(),
                    time=datetime.now().time(),
                    status="Present"
                )

                attendance_marked = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ⏱ AUTO STOP AFTER TIME LIMIT
        if time.time() - start_time > TIME_LIMIT:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

    camera.release()
    cv2.destroyAllWindows()

# =====================
# START ATTENDANCE
# =====================
def start_face_detection(request):
    global attendance_marked
    attendance_marked = False

    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
