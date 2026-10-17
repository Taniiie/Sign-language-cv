<div align="center">

<img src="assets/images/banner.png" alt="Sign Language CV Banner" width="100%" />

# Sign Language CV

### Real-Time Computer Vision & Deep Learning Sign Language Translator with Text-to-Speech

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/MediaPipe-Tasks_Vision-0078D4?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Audio-pyttsx3_TTS-06B6D4?style=for-the-badge&logo=soundcharts&logoColor=white" alt="TTS" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  An end-to-end, high-performance sign language recognition pipeline capable of tracking two hands, modeling temporal gesture dynamics across sequential frames with a deep LSTM network, and vocalizing detected phrases in real-time through an offline speech synthesis engine.
</p>

</div>

---

## <img src="assets/icons/sparkles.svg" width="22" height="22" align="center" /> Overview

**Sign Language CV** bridges the communication barrier between sign language users and non-signers. Unlike traditional image classification models that evaluate static single-frame gestures, this project captures the **temporal motion trajectory** of hand movements across a rolling window of 30 frames. 

By leveraging Google's **MediaPipe Hand Landmarker** vision pipeline, the system extracts 21 precise 3D landmark coordinates per hand (up to 2 hands = 126 coordinate values per frame). These temporal sequences are fed into a recurrent **Long Short-Term Memory (LSTM)** neural network to infer signs with high confidence before audibly announcing them with **pyttsx3**.

---

## <img src="assets/icons/check.svg" width="22" height="22" align="center" /> Key Highlights

- **<img src="assets/icons/camera.svg" width="18" height="18" align="center" /> Dual-Hand 3D Landmark Tracking:** Tracks up to 42 individual hand landmarks in real-time at 30+ FPS using modern MediaPipe Tasks Vision (`hand_landmarker.task`).
- **<img src="assets/icons/brain.svg" width="18" height="18" align="center" /> Temporal Sequence Modeling:** Deep 2-layer LSTM architecture with recurrent dropout layers capable of distinguishing dynamic multi-frame gestures from static hand poses.
- **<img src="assets/icons/speaker.svg" width="18" height="18" align="center" /> Instant Speech Feedback:** Integrated offline Text-to-Speech (TTS) engine that speaks phrases as soon as gesture prediction confidence exceeds the threshold (85%+).
- **<img src="assets/icons/database.svg" width="18" height="18" align="center" /> Custom Dataset Pipeline:** Includes built-in interactive dataset collection utilities with automatic on-screen countdowns and NumPy `.npy` sequential matrix storage.
- **<img src="assets/icons/settings.svg" width="18" height="18" align="center" /> Lightweight & Edge-Ready:** Runs entirely on CPU with zero cloud latency or subscription requirements.

---

## <img src="assets/icons/layers.svg" width="22" height="22" align="center" /> System Architecture & Pipeline

<div align="center">
  <img src="assets/images/pipeline_architecture.png" alt="Sign Language CV Architecture" width="100%" />
</div>

The real-time recognition pipeline operates in four orchestrated stages:

1. **Vision Ingestion & Landmark Extraction:**
   - Captures webcam stream at 720p / 30 FPS via OpenCV.
   - Converts frames to RGB and feeds them into `mp.tasks.vision.HandLandmarker`.
   - Normalizes and flattens $(x, y, z)$ coordinates for 2 hands (zero-padded when one hand is obscured):
     $$\text{Keypoints} \in \mathbb{R}^{2 \times 21 \times 3} = \mathbb{R}^{126}$$

2. **Rolling Window Sequence Buffer:**
   - Collects keypoint vectors into a rolling FIFO queue of $T = 30$ frames:
     $$\text{Input Tensor} \in \mathbb{R}^{1 \times 30 \times 126}$$

3. **LSTM Classification & Confidence Filtering:**
   - Evaluates the rolling sequence using the trained LSTM model.
   - Extracts probability distributions via Softmax:
     $$\hat{y} = \text{argmax}(\text{Softmax}(Z)), \quad \text{Condition: } P(\hat{y}) \ge 0.85$$

4. **Speech Synthesis Engine:**
   - Dispatches debounced label notifications to `pyttsx3` to articulate words aloud without stuttering or infinite loops.

---

## <img src="assets/icons/hand.svg" width="22" height="22" align="center" /> Supported Gestures

| Sign | Category | Action Type | Vocabulary Target |
| :--- | :--- | :--- | :--- |
| **Hello** | Greeting | Wave / Open Palm Forward | `hello` |
| **Thank You** | Courtesy | Hand to Chin Forward Extension | `thank_you` |
| **Yes** | Affirmation | Fist Nodding Downward | `yes` |
| **No** | Negation | Index & Middle to Thumb Pinch | `no` |
| **Please** | Politeness | Open Hand Circular Chest Motion | `please` |

*Easily expandable to additional ASL words, numbers, or phrases by adding labels to `collect_data.py`.*

---

## <img src="assets/icons/chart.svg" width="22" height="22" align="center" /> Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 30, 64)            48,896    
 dropout (Dropout)           (None, 30, 64)            0         
 lstm_1 (LSTM)               (None, 128)               98,816    
 dropout_1 (Dropout)         (None, 128)               0         
 dense (Dense)               (None, 64)                8,256     
 dense_1 (Dense)             (None, 5)                 325       
=================================================================
 Total params: 156,293 (610.52 KB)
 Trainable params: 156,293 (610.52 KB)
 Optimizer: Adam | Loss: Categorical Crossentropy | Metric: Accuracy
```

---

## <img src="assets/icons/folder.svg" width="22" height="22" align="center" /> Repository Structure

```
Sign-language-cv/
├── assets/
│   ├── icons/                  # Vector SVG icons for documentation
│   │   ├── brain.svg
│   │   ├── camera.svg
│   │   ├── hand.svg
│   │   └── ...
│   └── images/                 # Architecture infographics & banners
│       ├── banner.png
│       └── pipeline_architecture.png
├── data/                       # Recorded gesture coordinate sequences
│   ├── hello/                  # 30 sequence matrices (30 x 126 .npy)
│   ├── no/
│   ├── please/
│   ├── thank_you/
│   └── yes/
├── collect_data.py             # Interactive gesture recorder
├── hand_tracking.py            # Diagnostic script for MediaPipe landmarks
├── real_time_predict.py        # Live webcam translator with TTS vocalization
├── test_webcam.py              # Hardware webcam connectivity test
├── train_model.py              # Keras LSTM neural network training pipeline
├── hand_landmarker.task        # MediaPipe landmark model bundle
├── sign_model.h5               # Trained LSTM model weights
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## <img src="assets/icons/terminal.svg" width="22" height="22" align="center" /> Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Taniiie/Sign-language-cv.git
cd Sign-language-cv
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Camera & Hand Landmark Tracking
Run the baseline tracking script to ensure your webcam and MediaPipe model bundle are working properly:
```bash
python hand_tracking.py
```
*Press `q` on the video window to exit.*

---

## <img src="assets/icons/settings.svg" width="22" height="22" align="center" /> Workflow & Usage

### Step 1: Collect Custom Sign Sequences
To record your own hand gestures:
1. Open `collect_data.py` and specify the gesture label in `SIGNS = ['your_gesture']`.
2. Run the collection script:
```bash
python collect_data.py
```
3. The interface will provide a 1.5-second countdown window before each sequence so you can prepare your hand position. Sequences are saved as `.npy` matrices in the `data/` directory.

### Step 2: Train the LSTM Model
Train the deep neural network on the captured temporal sequences:
```bash
python train_model.py
```
This loads all `.npy` windows, performs train-test splitting (85/15), fits the LSTM network over 100 epochs, and serializes the resulting model to `sign_model.h5`.

### Step 3: Run Real-Time Translation & TTS
Execute live sign language inference:
```bash
python real_time_predict.py
```
- Green landmark nodes will trace finger joints and knuckles in real-time.
- When an identified sign crosses the confidence threshold (85%), the predicted label and score appear on screen, and your computer will speak the sign aloud.

---

## <img src="assets/icons/book.svg" width="22" height="22" align="center" /> Configuration Parameters

You can adjust hyperparameters in `real_time_predict.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `SEQUENCE_LENGTH` | `30` | Number of consecutive frames captured for each gesture decision window. |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum classification probability required before triggering speech. |
| `num_hands` | `2` | Maximum hands detected simultaneously by MediaPipe. |
| `speech_rate` | `150` | Word articulation speed (words per minute) configured in `pyttsx3`. |

---

## <img src="assets/icons/license.svg" width="22" height="22" align="center" /> License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Developed by <a href="https://github.com/Taniiie">Taniiie</a> with MediaPipe, TensorFlow, and OpenCV.</sub>
</div>
