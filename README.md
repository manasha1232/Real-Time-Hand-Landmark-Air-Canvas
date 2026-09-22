# 🎨 Real-Time Hand Landmark Air Canvas & Digital Painting Studio

[![Day](https://img.shields.io/badge/Day-24--30-blue?style=for-the-badge&logo=python)](https://github.com/manasha1232/30-Day-Computer-Vision-Challenge)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%203D-orange?style=for-the-badge&logo=google)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

An interactive **Computer Vision Air Canvas & Digital Painting System** that allows users to paint in 3D physical space using 21 MediaPipe skeletal hand landmarks and intuitive finger gestures. Features dynamic color palette selection, exponential moving average (EMA) stroke anti-jitter smoothing, pinch-controlled brush sizing, and a 2-panel HUD output montage.

---

## 🌟 Key Features

- 🖐️ **21 3D Skeletal Landmark Tracking**: Real-time hand landmark extraction using MediaPipe Hands.
- ✌️ **Gesture State Machine**:
  - **DRAW MODE**: Index finger extended up, middle finger folded down $\rightarrow$ Continuous smooth line painting.
  - **SELECTION MODE**: Both Index & Middle fingers extended $\rightarrow$ Hover cursor to choose colors or clear canvas.
  - **DYNAMIC BRUSH SIZE**: Distance between Thumb Tip (LM 4) & Index Tip (LM 8) controls line thickness ($3\text{px} - 30\text{px}$).
  - **CLEAR CANVAS**: Select the `CLEAR` button on top HUD banner or make a fist.
- 🎨 **Multi-Color Palette**: Blue, Green, Red, Yellow, Purple, Eraser, and Clear buttons.
- 📈 **EMA Anti-Jitter Filter**: Exponential Moving Average smoothing ($x_{\text{smooth}} = \alpha \cdot x_{\text{new}} + (1-\alpha) \cdot x_{\text{prev}}$) eliminates hand tremor noise.
- 📊 **Telemetry & Metric Audit**: Exports structured JSON logs detailing stroke counts, line distance in pixels, FPS performance, and color usage distribution.

---

## 🛠️ Installation & Setup

```bash
# Clone the project repository
git clone https://github.com/manasha1232/hand_landmark_air_canvas.git
cd hand_landmark_air_canvas

# Install required dependencies
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1️⃣ Run with Synthetic Video Generator (Default Batch Mode)
```bash
python generate_demo_air_canvas.py
python air_canvas.py
```

### 2️⃣ Run with Live Webcam Input
```bash
python air_canvas.py --source 0
```

---

## 📊 Sample Output Telemetry JSON

```json
{
    "project": "Real-Time Hand Landmark Air Canvas",
    "day": 24,
    "status": "SUCCESS",
    "resolution": {
        "width": 1280,
        "height": 720
    },
    "performance": {
        "total_frames_processed": 90,
        "execution_duration_sec": 2.14,
        "average_fps": 42.06
    },
    "canvas_metrics": {
        "total_drawing_strokes": 89,
        "total_distance_pixels": 1845.32,
        "color_usage_counts": {
            "BLUE": 30,
            "GREEN": 30,
            "RED": 29,
            "YELLOW": 0,
            "PURPLE": 0,
            "ERASER": 0
        },
        "gesture_mode_counts": {
            "DRAW": 89,
            "SELECTION": 0,
            "HOVER": 0
        }
    }
}
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
