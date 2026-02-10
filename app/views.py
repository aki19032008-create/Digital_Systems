import cv2
import face_recognition
import numpy as np
from datetime import date

from django.shortcuts import render
from django.http import StreamingHttpResponse

from .models import Student, Attendance


# --------------------------------------------------
# GLOBALS (demo / mini project purpose)
# --------------------------------------------------

camera = cv2.VideoCapture(0)

blink_count = 0
blink_done = False
recognized_student = None


# --------------------------------------------------
# NORMAL PAGES
# --------------------------------------------------

def home(request):
    return render(request, 'app/home.html')


def staff_login(request):
    return render(request, 'app/staff_login.html')


def attendance_register(request):
    """
    Page shows detected student details AFTER face + blink verified
    """
    return render(
        request,
        'app/attendance_register.html',
        {'student': recognized_student}
    )


# --------------------------------------------------
# EAR (Eye Aspect Ratio) – Blink Detection
# --------------------------------------------------

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


# --------------------------------------------------
# LOAD STUDENT FACE DATA (FROM DB)
# --------------------------------------------------

known_encodings = []
known_students = []

def load_student_faces():
    """
    Load all student images from DB
    """
    known_encodings.clear()
    known_students.clear()

    for student in Student.objects.all():
        if student.image:
            img = face_recognition.load_image_file(student.image.path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_students.append(student)

# load once when server starts
load_student_faces()


# --------------------------------------------------
# CAMERA STREAM (Blink + Face Recognition)
# --------------------------------------------------

def gen_frames():
    global blink_count, blink_done, recognized_student

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        landmarks = face_recognition.face_landmarks(rgb)

        # ---------------- BLINK CHECK ----------------
        for lm in landmarks:
            left_eye = np.array(lm['left_eye'])
            right_eye = np.array(lm['right_eye'])

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            if ear < 0.20:
                blink_count += 1

            if blink_count >= 3:
                blink_done = True

        # ---------------- FACE RECOGNITION ----------------
        if blink_done and face_locations:
            encodings = face_recognition.face_encodings(rgb, face_locations)

            for enc in encodings:
                matches = face_recognition.compare_faces(known_encodings, enc)

                if True in matches:
                    idx = matches.index(True)
                    recognized_student = known_students[idx]

                    # Save attendance (once per day)
                    Attendance.objects.get_or_create(
                        student=recognized_student,
                        date=date.today()
                    )

                    cv2.putText(
                        frame,
                        f"{recognized_student.name} - Present",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # ---------------- STREAM FRAME ----------------
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )


# --------------------------------------------------
# STREAM VIEW
# --------------------------------------------------

def start_face_detection(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
