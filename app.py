from flask import Flask, render_template, Response, jsonify, request
import cv2
import random
import mediapipe as mp
import os
import numpy as np
from collections import defaultdict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== LOAD MODELS ==================

age_net = cv2.dnn.readNetFromCaffe(
    r'C:\Users\samrah\OneDrive\Desktop\AI facial & health insight\models\age_deploy.prototxt',
    r'C:\Users\samrah\OneDrive\Desktop\AI facial & health insight\models\age_net.caffemodel'
)

gender_net = cv2.dnn.readNetFromCaffe(
    r'C:\Users\samrah\OneDrive\Desktop\AI facial & health insight\models\gender_deploy.prototxt',
    r'C:\Users\samrah\OneDrive\Desktop\AI facial & health insight\models\gender_net.caffemodel'
)

smile_cascade = cv2.CascadeClassifier(
    r'C:\Users\samrah\OneDrive\Desktop\AI facial & health insight\models\haarcascade_smile.xml'
)

print("Age net loaded:", age_net.empty())
print("Gender net loaded:", gender_net.empty())

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_list = ['Male', 'Female']

# ================== MEDIAPIPE ==================

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# ================== GLOBAL STATE ==================

current_results = []
emotion_count = defaultdict(int)

last_gender = None
last_age = None

# ================== EMOTION ==================

def detect_emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    smiles = smile_cascade.detectMultiScale(gray, 1.7, 20)

    h, w = gray.shape
    mouth = gray[int(h*0.6):h, int(w*0.2):int(w*0.8)]
    variance = np.var(mouth)

    if len(smiles) > 0:
        return "Happy", 90
    elif variance < 180:
        return "Sad", 80
    elif variance > 400:
        return "Stressed", 82
    else:
        return "Neutral", 75

# ================== AI RESPONSE ==================

def generate_ai_advice(age, emotion, confidence):
    if emotion == "Happy":
        return f"You seem happy ({confidence:.1f}%). Keep smiling 😊"
    elif emotion == "Sad":
        return f"You look low ({confidence:.1f}%). Try relaxing."
    elif emotion == "Stressed":
        return f"Stress detected ({confidence:.1f}%). Take a break."
    else:
        return f"You appear calm ({confidence:.1f}%). Stay focused."

# ================== PROCESS FACE ==================

def process_face(face):
    global last_gender, last_age

    if face is None or face.size == 0:
        return "Unknown", "Unknown", "Neutral", 0, "", ""

    try:
        face = cv2.resize(face, (227, 227))
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                     (78.426,87.768,114.895))

        # Gender
        gender_net.setInput(blob)
        gender_pred = gender_list[gender_net.forward().argmax()]
        gender = last_gender if last_gender else gender_pred
        last_gender = gender

        # Age
        age_net.setInput(blob)
        age_pred = age_list[age_net.forward().argmax()]
        age = last_age if last_age else age_pred
        last_age = age

    except Exception as e:
        print("Model error:", e)
        return "Unknown", "Unknown", "Neutral", 0, "", ""

    emotion, confidence = detect_emotion(face)
    advice = generate_ai_advice(age, emotion, confidence)

    return gender, age, emotion, confidence, advice, ""

# ================== VIDEO ==================

def generate_frames():
    global current_results

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)
        face_results = []

        if results.detections:
            for i, detection in enumerate(results.detections):

                bbox = detection.location_data.relative_bounding_box

                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                if bw < 80 or bh < 80:
                    continue

                face = frame[y:y+bh, x:x+bw]

                gender, age, emotion, confidence, advice, _ = process_face(face)

                face_results.append({
                    "id": f"Person {i+1}",
                    "gender": gender,
                    "age": age,
                    "emotion": emotion,
                    "confidence": confidence,
                    "advice": advice
                })

                cv2.rectangle(frame, (x,y),(x+bw,y+bh),(0,255,0),2)

        if face_results:
            current_results = face_results

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================== ROUTES ==================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    return jsonify(current_results)

@app.route('/upload', methods=['POST'])
def upload():
    global current_results

    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)
    face_results = []

    if results.detections:
        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * image.shape[1])
            y = int(bbox.ymin * image.shape[0])
            w = int(bbox.width * image.shape[1])
            h = int(bbox.height * image.shape[0])

            face = image[y:y+h, x:x+w]

            gender, age, emotion, confidence, advice, _ = process_face(face)

            face_results.append({
                "id": f"Person {i+1}",
                "gender": gender,
                "age": age,
                "emotion": emotion,
                "confidence": confidence,
                "advice": advice
            })

    current_results = face_results
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)