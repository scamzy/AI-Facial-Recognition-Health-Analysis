# 🚀 AI Driven Facial Recognition & Health Analysis

## 📌 Overview

This project is a real-time AI-based system that performs facial detection and basic health analysis using computer vision and deep learning.
It detects faces via webcam or uploaded images, predicts age and gender, estimates emotion, and provides simple health insights through a web interface.

---

## 🎯 Features

* 📷 Real-time face detection using MediaPipe
* 👤 Age & Gender prediction using Caffe models
* 😊 Emotion analysis (lightweight logic)
* 💡 Health advice generation
* 🌐 Flask-based web application
* 📤 Image upload support
* 🕘 History tracking

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* Flask
* Caffe (Deep Learning Models)

---

## ⚙️ Setup & Implementation Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/scamzy/AI-Facial-Recognition-Health-Analysis.git
cd ai-facial-Recognition-Health-Analysis
```

---

### 2️⃣ Install Dependencies

```bash
pip install opencv-python mediapipe flask numpy
```

---

### 3️⃣ Download Required Models

Download the following Caffe model files and place them in the project folder:

* age_deploy.prototxt
* age_net.caffemodel
* gender_deploy.prototxt
* gender_net.caffemodel

👉 Add your download links here:

* Age Model: https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0
* Gender Model: https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0

---

### 4️⃣ Run the Application

```bash
python app.py
```

---

### 5️⃣ Open in Browser

http://127.0.0.1:5000

---

## ⚠️ Limitations

* Accuracy depends on lighting conditions
* Age prediction is approximate
* Emotion detection is simulated

---

## 🔮 Future Enhancements

* LLM integration (Gemini / GPT)
* Improved emotion detection
* Database for storing history

---

## 👨‍💻 Author

**Samrah Inayathulla**
Final Year BE Project
