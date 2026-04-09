from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import random
import time
import mediapipe as mp
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt','age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt','gender_net.caffemodel')

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_list = ['Male', 'Female']
emotions = ["Happy", "Sad", "Stressed", "Neutral"]

# MediaPipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# Global state
last_prediction_time = 0
prediction_interval = 6  # 🔥 Increased stability

current_gender = ""
current_age = ""
current_emotion = ""
current_advice = "Waiting for detection..."

history = []

def get_health_advice(age, emotion):
    if emotion == "Stressed":
        return "Stress detected. Take breaks and relax."
    elif emotion == "Sad":
        return "Low mood detected. Talk to friends and rest."
    elif emotion == "Happy":
        return "Positive mood. Maintain your lifestyle."
    else:
        return "Stay healthy with proper diet and exercise."

def process_face(face):
    global current_emotion

    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227),
                                 (78.426,87.768,114.895))

    # Gender
    gender_net.setInput(blob)
    gender = gender_list[gender_net.forward().argmax()]

    # Age
    age_net.setInput(blob)
    age = age_list[age_net.forward().argmax()]

    # 🔥 Emotion stabilization
    if current_emotion == "":
        emotion = random.choice(emotions)
    else:
        emotion = current_emotion

    advice = get_health_advice(age, emotion)

    return gender, age, emotion, advice

def generate_frames():
    global last_prediction_time, current_gender, current_age, current_emotion, current_advice

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Brightness fix
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=40)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face = frame[y:y+bh, x:x+bw]

                current_time = time.time()

                if face.size > 0 and current_time - last_prediction_time > prediction_interval:

                    gender, age, emotion, advice = process_face(face)

                    current_gender = gender
                    current_age = age
                    current_emotion = emotion
                    current_advice = advice

                    history.append({
                        "gender": gender,
                        "age": age,
                        "emotion": emotion
                    })

                    last_prediction_time = current_time

                cv2.rectangle(frame, (x,y),(x+bw,y+bh),(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    return jsonify({
        "gender": current_gender,
        "age": current_age,
        "emotion": current_emotion,
        "advice": current_advice
    })

@app.route('/history')
def get_history():
    return jsonify(history[-5:])

@app.route('/upload', methods=['POST'])
def upload():
    global current_gender, current_age, current_emotion, current_advice

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=40)

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            face = image[y:y+bh, x:x+bw]

            if face.size > 0:
                gender, age, emotion, advice = process_face(face)

                current_gender = gender
                current_age = age
                current_emotion = emotion
                current_advice = advice

                history.append({
                    "gender": gender,
                    "age": age,
                    "emotion": emotion
                })

    return "OK"

if __name__ == "__main__":
    app.run(debug=True)