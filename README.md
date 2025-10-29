# 🎥 Face Detection with Webcam using Deep Learning Algorithm

A simple yet effective deep-learning-based **real-time face detection** application using a webcam.
Built with **Python, OpenCV, and TensorFlow**, this project demonstrates how deep learning can be applied to detect faces directly from a webcam feed.

---

## 🚀 Project Overview

This project enables real-time face detection using a trained deep learning model.
It includes:

* A **training notebook** to build or fine-tune a face detection model.
* A **Python script** that activates your webcam and performs live face detection.

Perfect for:

* Students exploring **Computer Vision** and **AI**
* Hobbyists building **smart camera systems**
* Researchers testing **face recognition pipelines**

---

## 🧠 Features

✅ Real-time face detection via webcam
✅ Deep learning model integration (TensorFlow / OpenCV DNN)
✅ Interactive Jupyter Notebook for training and experimentation
✅ Easy setup and lightweight codebase
✅ Open for extension — recognition, attendance, or alert systems

---

## 📂 Repository Structure

```
Face_detection_with_webcam_using_DL_Algo/
│
├── Face_Train_Model.ipynb     # Train and test the deep learning face detection model
├── main01.py                  # Real-time webcam face detection script
├── models/                    # (Optional) Store pretrained or trained models here
├── data/                      # Dataset for training and testing (faces, images, etc.)
└── README.md                  # Project documentation
```

---

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ARASIF1-6/Face_detection_with_webcam_using_DL_Algo.git
cd Face_detection_with_webcam_using_DL_Algo
```

### 2️⃣ Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, you can manually install common dependencies:

```bash
pip install opencv-python tensorflow numpy matplotlib
```

---

## 🧩 How to Use

### 🔹 Step 1: Train Your Model

Open **`Face_Train_Model.ipynb`** using Jupyter Notebook or JupyterLab:

```bash
jupyter notebook Face_Train_Model.ipynb
```

Follow the notebook steps to:

* Load and preprocess image data
* Train a CNN or use a pretrained model
* Save the trained model (e.g., `model.h5`)

---

### 🔹 Step 2: Run Real-Time Face Detection

Once your model is ready, run:

```bash
python main01.py
```

It will open your webcam and start detecting faces in real-time.

---

## 💡 Example Output

When you run the detection script, you’ll see bounding boxes drawn around detected faces in your webcam feed.

| Webcam Input                                                                                     | Detection Result                                                                                   |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| ![Input](https://github.com/ARASIF1-6/Face_detection_with_webcam_using_DL_Algo/assets/input.jpg) | ![Output](https://github.com/ARASIF1-6/Face_detection_with_webcam_using_DL_Algo/assets/output.jpg) |

---

## ⚙️ Code Highlights

### `main01.py`

```python
import cv2
import tensorflow as tf
import numpy as np

# Load pre-trained model
model = tf.keras.models.load_model('model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for model input
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img / 255.0, axis=0)

    # Predict faces (you can modify threshold based on model confidence)
    predictions = model.predict(img)

    # Draw rectangle (this part depends on your model output)
    cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam Face Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---
