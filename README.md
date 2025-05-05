## 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **Python**, **OpenCV**, and **MediaPipe**, designed to monitor eye aspect ratio (EAR) and mouth aspect ratio (MAR) to detect signs of fatigue like eye closure and yawning.

---

### 🔍 Features

* Real-time detection using webcam
* Calculates **EAR** for eye closure detection
* Calculates **MAR** for yawning detection
* Displays EAR and MAR values on screen
* Alerts when drowsiness is detected

---

### 🧠 How It Works

* **EAR (Eye Aspect Ratio)**: Measures vertical vs. horizontal eye opening. A low EAR for consecutive frames indicates drowsiness.
* **MAR (Mouth Aspect Ratio)**: Measures vertical mouth opening to detect yawns.

Thresholds:

* EAR < 0.25 for 20+ frames → Drowsiness alert
* MAR > 0.75 for 15+ frames → Yawning alert

---

### 💠 Requirements

* Python 3.7+
* OpenCV
* MediaPipe
* NumPy
* SciPy

Install dependencies using:

```bash
pip install opencv-python mediapipe numpy scipy
```

---

### ▶️ How to Run

```bash
python "driver drowziness detection.py"
```

Press **'q'** to quit the detection window.

---

### 🖼️ Output Screenshot

Here’s a sample output where drowsiness is detected in real-time with EAR and MAR displayed:

![Drowsiness Detection Output](https://i.imgur.com/0dRBpR3.png)
*If you want me to generate and upload a real-time image output using a simulation, let me know!*

---

### 📂 File Structure

```
📁 Driver-Drowsiness-Detection
│
├— driver drowziness detection.py
├— README.md
```

---

### 📌 Future Improvements

* Add sound alert (e.g., using `playsound`)
* Export logs of drowsiness detection
* Deploy to mobile with TensorFlow Lite
* Integrate with vehicle systems

---

### 🧑‍💻 Author

M.Sandeep Sagar
www.linkedin.com/in/madanu-sandeep-sagar
SandeepSagarMadanu(github)
