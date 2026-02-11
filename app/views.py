import cv2
import mediapipe as mp
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse

# --------------------------------------------------
# GLOBALS
# --------------------------------------------------

camera = cv2.VideoCapture(0)

blink_count = 0
blink_done = False
face_detected = False

# DEFAULT STUDENT (TEMP – no DB, no CSV)
recognized_student = None
DEFAULT_STUDENT = {
    "roll": "101",
    "name": "Arun",
    "status": "Present"
}

# --------------------------------------------------
# MEDIAPIPE SETUP
# --------------------------------------------------

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_mesh.FaceMesh(max_num_faces=1)

# --------------------------------------------------
# EAR (Blink Detection)
# --------------------------------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_points):
    p = landmarks
    A = np.linalg.norm(p[eye_points[1]] - p[eye_points[5]])
    B = np.linalg.norm(p[eye_points[2]] - p[eye_points[4]])
    C = np.linalg.norm(p[eye_points[0]] - p[eye_points[3]])
    return (A + B) / (2.0 * C)

# --------------------------------------------------
# PAGES
# --------------------------------------------------

def home(request):
    return render(request, 'app/home.html')

def staff_login(request):
    return render(request, 'app/staff_login.html')

def attendance_register(request):
    return render(
        request,
        'app/attendance_register.html',
        {'student': recognized_student}
    )

# --------------------------------------------------
# CAMERA STREAM (Face + Blink)
# --------------------------------------------------

def gen_frames():
    global blink_count, blink_done, face_detected, recognized_student

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # ---------- FACE DETECTION ----------
        face_results = face_detection.process(rgb)

        if face_results.detections:
            face_detected = True

            for det in face_results.detections:
                box = det.location_data.relative_bounding_box
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        else:
            face_detected = False
            blink_count = 0
            blink_done = False
            recognized_student = None

        # ---------- BLINK CHECK ----------
        mesh_results = face_mesh.process(rgb)

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = np.array(
                    [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                )

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2

                if ear < 0.20:
                    blink_count += 1

                if blink_count >= 2:
                    blink_done = True
                    recognized_student = DEFAULT_STUDENT

        # ---------- DISPLAY ----------
        if blink_done:
            cv2.putText(
                frame,
                "Liveness Verified",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

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
