Forehead tracker computer vision, could be used for posture tracking and more. 

A real-time Computer Vision application that utilizes **Google MediaPipe Face Mesh** to track specific facial landmarks for posture analysis.

Key Features
High-Precision Tracking: Utilizes Landmark #10 (Forehead Center) for calculating head tilt and posture deviation.
Gesture Control: Implements a custom state machine to toggle the UI overlay by detecting mouth-open gestures (calculating Euclidean distance between lip landmarks).
Debouncing Logic: Includes threshold-based signal debouncing to prevent gesture flickering.Apple Silicon Optimization: specifically engineered with `startWindowThread` and permission handling to run natively on macOS M-series chips without latency.

## 🛠️ Technical Stack
Language: Python 3.11
Vision:  OpenCV (cv2)
AI Model: MediaPipe Face Mesh (Dense 468-point mesh)
Hardware: Optimized for Mac Webcam (Index Switching Support)

📦 How to Run
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
