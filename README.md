# 🤖 MediaPipe Computer Vision Projects

A collection of Python projects using **Google's MediaPipe** framework for real-time AI-powered computer vision tasks — including object detection from images and live 3D face landmark transformation via webcam.

---

## 📁 Project Structure

```
mediapipe_project/
│
├── code1.py                    # Object Detection from Image URL
├── code2.py                    # Real-Time 3D Face Landmark Detection (Webcam)
├── efficientdet_lite0.tflite   # Object detection model (auto-downloaded)
├── face_landmarker.task        # Face landmarker model (auto-downloaded)
└── README.md                   # Project documentation
```

---

## 📌 Project 1 — Object Detection from Image URL (`code1.py`)

### 🔍 What It Does

This script fetches an image from a given URL and uses **MediaPipe's EfficientDet Lite** model to detect objects in the image. Detected objects are highlighted with **green bounding boxes** and labeled with their **category name and confidence score**, then displayed using Matplotlib.

### 🛠️ How It Works

1. **Downloads the model** (`efficientdet_lite0.tflite`) automatically if not present locally.
2. **Fetches the image** from a public URL using `urllib` and converts it to an RGB NumPy array.
3. **Creates an ObjectDetector** using MediaPipe Tasks API with a confidence threshold of `0.2`.
4. **Runs inference** on the image and retrieves detection results.
5. **Draws bounding boxes** and category labels on the image using OpenCV.
6. **Displays the annotated image** via Matplotlib.

### ⚙️ Key Configuration

| Parameter         | Value                          |
|------------------|-------------------------------|
| Model            | EfficientDet Lite 0 (INT8)     |
| Score Threshold  | 0.2 (20% confidence minimum)   |
| Image Source     | Remote URL (configurable)      |
| Output           | Annotated image (Matplotlib)   |

### 📦 Requirements

```bash
pip install mediapipe opencv-python numpy matplotlib
```

### ▶️ Run

```bash
python code1.py
```

> 💡 You can change the `mug` variable in the script to any public image URL to detect objects in a different image.

---

## 📌 Project 2 — Real-Time 3D Face Landmark Detection (`code2.py`)

### 🔍 What It Does

This script uses your **live webcam feed** to detect a face and map **468 3D facial landmarks** onto it in real time using MediaPipe's **Face Landmarker** model. It also applies a **3D transformation (scaling)** to the landmark coordinates and visualizes both the raw mesh and the transformed points simultaneously.

### 🛠️ How It Works

1. **Downloads the model** (`face_landmarker.task`) automatically if not present locally.
2. **Opens the webcam** using OpenCV (`cv2.VideoCapture(0)`).
3. **Flips the frame** horizontally for a natural mirror effect.
4. **Sends each frame** asynchronously to the Face Landmarker in `LIVE_STREAM` mode.
5. **Applies a 3D transformation matrix** (1.5x scaling) to the detected landmark coordinates.
6. **Draws the transformed points** (blue dots) and the original face mesh connections (green dots) on the frame.
7. **Displays the result** in a live OpenCV window — press `Q` to quit.

### ⚙️ Key Configuration

| Parameter                    | Value                          |
|-----------------------------|-------------------------------|
| Model                       | Face Landmarker (Float16)      |
| Running Mode                | LIVE_STREAM (async)            |
| Max Faces                   | 1                              |
| Min Detection Confidence    | 0.5 (50%)                      |
| Transformation              | 3D scaling (1.5x in x & y)     |
| Output                      | Live OpenCV window             |

### 📦 Requirements

```bash
pip install mediapipe opencv-python numpy
```

### ▶️ Run

```bash
python code2.py
```

> 💡 Press **`Q`** on your keyboard to quit the webcam window.
>
> 🎥 Make sure your webcam is connected and not being used by another application.

---

## 🧠 What is MediaPipe?

**MediaPipe** is an open-source, cross-platform framework developed by **Google** for building real-time, on-device machine learning pipelines. It is widely used for computer vision and AI tasks directly on the device — without needing a cloud server.

### ✨ Key Features

- **Real-time performance** — optimized for live video/image processing
- **Cross-platform** — runs on Python, Android, iOS, Web, and Raspberry Pi
- **Pre-trained models** — comes with production-ready models for common tasks
- **On-device inference** — no internet connection required once models are downloaded
- **MediaPipe Tasks API** — a simple, high-level API for integrating AI into any project

---

## 🚀 Projects You Can Build with MediaPipe

| Category              | Project Ideas                                                                 |
|----------------------|-------------------------------------------------------------------------------|
| 👁️ Face & Head       | Face mesh overlay, emotion detector, drowsiness/eye-blink detector, AR filters |
| 🖐️ Hand & Gesture    | Hand gesture controller, sign language translator, virtual mouse/keyboard     |
| 🧍 Body & Pose       | Yoga/exercise form checker, sports motion analyzer, fall detection system     |
| 🎯 Object Detection  | Real-time object counter, inventory tracker, smart security camera            |
| 🖼️ Image Segmentation| Background remover (like Zoom/Meet), portrait blur effect                     |
| 🧠 Face Recognition  | Attendance system, age/gender estimator, face authentication                  |
| 🤳 Augmented Reality | Virtual try-on (glasses, hats), face filters, makeup effects                  |
| 🎮 Interactive Apps  | Hands-free game controller, air-drawing canvas, music gesture controller      |

---

## 🔧 Dependencies Summary

| Library       | Purpose                                      |
|--------------|----------------------------------------------|
| `mediapipe`  | Core AI/ML models and inference engine        |
| `opencv-python` | Webcam capture, image drawing, display     |
| `numpy`      | Array manipulation and math operations        |
| `matplotlib` | Static image display (used in code1.py)       |
| `urllib`     | Fetching images and models from URLs          |

Install all at once:

```bash
pip install mediapipe opencv-python numpy matplotlib
```

---

## 📝 Notes

- Both scripts **auto-download** their required `.tflite` / `.task` model files on first run.
- For `code2.py`, ensure your system has a working webcam (device index `0` by default).
- Python **3.8 – 3.11** is recommended for full MediaPipe compatibility.
- MediaPipe models run fully **on-device** — no API key or internet is needed after download.

---

## 👨‍💻 Author

**Akhil**
Built with ❤️ using [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) by Google
