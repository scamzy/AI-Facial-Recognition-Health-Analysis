# 🚀 AI Driven Facial Recognition & Health Analysis

## 📌 Overview

This project is a real-time AI-based system that performs facial detection and basic health analysis using computer vision.

It detects faces via webcam or uploaded images, predicts age and gender using deep learning models, estimates emotion, and provides simple AI-based health advice.

---

## 🎯 Features

- 📷 Real-time face detection (OpenCV)
- 👤 Age & Gender prediction (Caffe models)
- 😊 Emotion detection (smile + eye based logic)
- 💡 AI-like health advice
- 📊 Session summary (emotion tracking)
- 📤 Image upload support
- 🌐 Flask web interface

---

## 🛠️ Tech Stack

- Python  
- OpenCV  
- Flask  
- NumPy  
- Caffe (Pre-trained models)

---

## ⚙️ Setup & Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/scamzy/AI-Facial-Recognition-Health-Analysis.git
cd AI-Facial-Recognition-Health-Analysis
```
### 2️⃣ Install dependencies
```bash
pip install opencv-python flask numpy
```

### 3️⃣ Add model files (IMPORTANT)

Download and place inside /models folder:

-Age Model
  Prototxt: https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt
  Caffemodel: https://www.dropbox.com/scl/fi/erfljon3f1chfyhbjdvoz/age_net.caffemodel?rlkey=4enbc7j5e138d8unnp8fz95p0&e=1&dl=0
-Gender Model
  Prototxt: https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt
  Caffemodel:https://www.dropbox.com/scl/fi/j7gcbj1l3ur6r8jdf3k7k/gender_net.caffemodel?rlkey=g7us6rvnw8ji7z64d5zy002h4&e=1&dl=0
Haarcascade (OpenCV)
Eye: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
Smile: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml

### 4️⃣ Run the app
```bash
python app.py
```

### 5️⃣ Open in browser
```bash
http://127.0.0.1:5000
```
## 3️⃣ Add model files (IMPORTANT)
Samrah Inayathulla
