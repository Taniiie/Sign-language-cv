import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
from tensorflow.keras.models import load_model

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)
landmarker = HandLandmarker.create_from_options(options)

model = load_model('sign_model.h5')
SIGNS = ['hello', 'thank_you', 'yes', 'no', 'please']

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85

def extract_keypoints(result):
    keypoints = np.zeros(2 * 21 * 3)
    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks[:2]):
            for j, lm in enumerate(hand):
                idx = (i * 21 + j) * 3
                keypoints[idx:idx+3] = [lm.x, lm.y, lm.z]
    return keypoints

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

sequence = []
current_prediction = ""
current_confidence = 0.0
last_spoken = ""

cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    keypoints = extract_keypoints(result)
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:]  # keep only the last 30 frames

    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(sequence, axis=0)  # shape (1, 30, 126)
        prediction = model.predict(input_data, verbose=0)[0]
        pred_idx = np.argmax(prediction)
        confidence = prediction[pred_idx]

        if confidence > CONFIDENCE_THRESHOLD:
            current_prediction = SIGNS[pred_idx]
            current_confidence = confidence

            if current_prediction != last_spoken:
                speak(current_prediction.replace('_', ' '))
                last_spoken = current_prediction

    # Draw hand keypoints
    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand in result.hand_landmarks:
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Display prediction
    if current_prediction:
        text = f'{current_prediction} ({current_confidence:.2f})'
        cv2.putText(frame, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    cv2.imshow('Sign Language Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()