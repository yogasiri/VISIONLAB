# VISIONLAB

## Computer Vision Algorithm Laboratory

VISIONLAB is an interactive educational laboratory designed to demonstrate and explore important Computer Vision and Face Analysis algorithms through a professional web interface.

The project allows users to understand how classical and deep-learning-based computer vision techniques work through interactive demonstrations, visual results, metrics, and explanations.

---

## 🚀 Live Demo

### 🌐 Try VISIONLAB Online

https://visionlab-nrwubrtihijizwebqnk9xf.streamlit.app/

No installation is required to try the deployed application.

---

## 🎯 Algorithms Demonstrated

VISIONLAB currently demonstrates four major Computer Vision techniques:

### 1. Template Matching

Demonstrates how OpenCV template matching can locate a smaller template image inside a larger image.

**Technology:**
- OpenCV
- `cv2.matchTemplate()`
- `cv2.minMaxLoc()`

**Displays:**
- Matching result
- Similarity score
- Match location
- Template dimensions

---

### 2. Viola-Jones

Demonstrates classical real-time face detection using Haar Cascade classifiers.

**Technology:**
- OpenCV
- Haar Cascade
- `CascadeClassifier()`

**Pipeline:**

```text
Haar-like Features
        ↓
Integral Image
        ↓
AdaBoost
        ↓
Cascade Classifier
        ↓
Face Detection
