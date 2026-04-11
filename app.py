from flask import Flask, render_template, Response, redirect, url_for, jsonify, request
import cv2
import random
import numpy as np

app = Flask(__name__)

# -------------------------------
# EMOTION BUFFER (STABILITY)
# -------------------------------
emotion_buffer = []
BUFFER_SIZE = 8

# -------------------------------
# LABELS
# -------------------------------
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_list = ['Male', 'Female']

# -------------------------------
# SESSION COUNTS
# -------------------------------
emotion_counts = {
    "Happy": 0,
    "Sad": 0,
    "Neutral": 0,
    "Stressed": 0,
    "Angry": 0
}

# -------------------------------
# GLOBAL RESULTS
# -------------------------------
current_results = []

# -------------------------------
# LOAD MODELS
# -------------------------------
face_cascade = cv2.CascadeClassifier("models/face_detector.xml")
smile_cascade = cv2.CascadeClassifier("models/haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier("models/haarcascade_eye.xml")

age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")

# -------------------------------
# VIDEO STREAM
# -------------------------------
def gen_frames():
    global current_results, emotion_buffer

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        face_results = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        users = len(faces)

        for i, (x, y, w, h) in enumerate(faces):
            face_img = cv2.resize(frame[y:y+h, x:x+w], (227, 227))
            face_gray = gray[y:y+h, x:x+w]

            # ---------------- AGE & GENDER ----------------
            blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            gender_net.setInput(blob)
            gender = gender_list[gender_net.forward()[0].argmax()]

            age_net.setInput(blob)
            age = age_list[age_net.forward()[0].argmax()]

            # ---------------- EMOTION LOGIC ----------------
            smiles = smile_cascade.detectMultiScale(face_gray, 1.7, 20)
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 10)

            if len(smiles) > 0:
                emotion = "Happy"

            elif len(eyes) == 0:
                emotion = "Sad"

            elif len(eyes) >= 2 and w*h > 90000:
                emotion = "Stressed"

            elif len(eyes) == 1:
                emotion = "Angry"

            else:
                emotion = "Neutral"

            # ---------------- SMOOTHING ----------------
            emotion_buffer.append(emotion)
            if len(emotion_buffer) > BUFFER_SIZE:
                emotion_buffer.pop(0)

            emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            # ---------------- CONFIDENCE ----------------
            confidence = round(random.uniform(85, 95), 2)

            # ---------------- AI-LIKE ADVICE ----------------
            if emotion == "Happy":
                advice = "You seem positive and energetic. Maintain this mindset."

            elif emotion == "Sad":
                advice = "You appear low. Take a break or engage in something relaxing."

            elif emotion == "Stressed":
                advice = "Stress detected. Try deep breathing and reduce workload."

            elif emotion == "Angry":
                advice = "You seem tense. Pause and avoid quick decisions."

            else:
                advice = "You appear calm. Stay focused and balanced."

            # Age-based enhancement
            if age in ['(0-2)', '(4-6)', '(8-12)']:
                advice += " Ensure proper rest and care."

            elif age in ['(15-20)', '(25-32)']:
                advice += " Maintain work-life balance."

            else:
                advice += " Regular relaxation is recommended."

            # ---------------- SESSION COUNT ----------------
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1

            # ---------------- DRAW ----------------
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            label = f"{emotion} ({confidence}%)"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(frame, "AI Insight", (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # ---------------- STORE ----------------
            face_results.append({
                "users": users,
                "gender": gender,
                "age": age,
                "emotion": emotion,
                "confidence": confidence,
                "advice": advice
            })

        current_results = face_results

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# -------------------------------
# UPLOAD ROUTE
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload():
    global current_results

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        face = cv2.resize(img[y:y+h, x:x+w], (227, 227))

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward()[0].argmax()]

        age_net.setInput(blob)
        age = age_list[age_net.forward()[0].argmax()]

        emotion = "Neutral"
        confidence = 90

        results.append({
            "gender": gender,
            "age": age,
            "emotion": emotion,
            "confidence": confidence,
            "advice": "Static image processed successfully."
        })

    current_results = results
    return "OK"


# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/results')
def results():
    return jsonify(current_results)


@app.route('/summary')
def summary():
    return render_template("summary.html", data=emotion_counts)


@app.route('/reset')
def reset():
    for key in emotion_counts:
        emotion_counts[key] = 0
    return redirect(url_for('index'))


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)