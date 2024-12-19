import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

model = load_model("new_face_train_model.h5")
class_names = ['Pappo Khan', 'Asif', 'AR Asif', 'Tanvir', 'Unknown']

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

cap = cv2.VideoCapture(0)

attendance = {}
confidence_threshold = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)

            face_image = frame[y:y + h, x:x + w]

            if face_image.size == 0:
                continue

            face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            face_image_resized = cv2.resize(face_image_gray, (160, 160))
            face_image_resized = np.expand_dims(face_image_resized, axis=-1)  # Add channel dimension
            face_image_resized = np.expand_dims(face_image_resized, axis=0)  # Add batch dimension
            face_image_resized = face_image_resized.astype('float32') / 255.0

            predictions = model.predict(face_image_resized)
            label_index = np.argmax(predictions)
            confidence = predictions[0][label_index]

            if confidence < confidence_threshold:
                name = "Unknown"
            else:
                name = class_names[label_index]

            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if name != "Unknown" and name not in attendance:
                attendance[name] = "Present"
                print(f"Attendance marked for {name}")

    present_text = ", ".join([f"{name} (Present)" for name in attendance.keys()])
    cv2.putText(frame, present_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Face Recognition Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance process completed.")
