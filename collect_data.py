import cv2
import mediapipe as mp
import numpy as np
import os
import time

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

# ---- CONFIG ----
SIGNS = ['no']
SEQUENCES_PER_SIGN = 30  # how many samples per sign
SEQUENCE_LENGTH = 30         # how many frames per sample
DATA_PATH = 'data'
# ----------------

for sign in SIGNS:
    os.makedirs(os.path.join(DATA_PATH, sign), exist_ok=True)

cap = cv2.VideoCapture(0)
start_time = time.time()

def extract_keypoints(result):
    # 2 hands x 21 landmarks x (x, y, z) = 126 values, zero-padded if a hand is missing
    keypoints = np.zeros(2 * 21 * 3)
    if result.hand_landmarks:
        for i, hand in enumerate(result.hand_landmarks[:2]):
            for j, lm in enumerate(hand):
                idx = (i * 21 + j) * 3
                keypoints[idx:idx+3] = [lm.x, lm.y, lm.z]
    return keypoints

for sign in SIGNS:
    for seq in range(SEQUENCES_PER_SIGN):
        print(f"\nGet ready for '{sign}' - sequence {seq+1}/{SEQUENCES_PER_SIGN}")
        window = []

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            keypoints = extract_keypoints(result)
            window.append(keypoints)

            # Countdown before first frame of each sequence
            if frame_num == 0:
                cv2.putText(frame, f'Starting: {sign} ({seq+1}/{SEQUENCES_PER_SIGN})',
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1500)  # 1.5 sec pause to get ready
            else:
                cv2.putText(frame, f'{sign} - frame {frame_num}/{SEQUENCE_LENGTH}',
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(1)

        window = np.array(window)
        save_path = os.path.join(DATA_PATH, sign, f'{seq}.npy')
        np.save(save_path, window)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
print("Data collection complete!")