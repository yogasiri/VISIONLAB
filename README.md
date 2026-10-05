# VISIONLAB
## Computer Vision Algorithm Laboratory

VisionLab is an educational Streamlit application demonstrating four computer-vision approaches:

- Template Matching
- Viola-Jones
- DeepFace
- FaceNet (Facenet512 through DeepFace)

## Features
- Professional dark multi-page UI
- Working OpenCV Template Matching
- Working Haar Cascade face detection
- Optional real DeepFace analysis and verification
- Optional real FaceNet-compatible embeddings
- Demo Gallery with prepared images
- Graceful handling of missing models/dependencies

## Technology Stack
Python, Streamlit, OpenCV, NumPy, Pillow, DeepFace.

## Project Structure
```text
VisionLab/
├── app.py
├── algorithms/
│   ├── template_matching.py
│   ├── viola_jones.py
│   ├── deepface.py
│   └── facenet.py
├── demo_images/
├── assets/
├── requirements.txt
└── README.md
```

## Installation
Use Python 3.10–3.12 for the smoothest DeepFace/TensorFlow compatibility.

```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```

## Demo
Open **Demo Gallery** and launch an algorithm. Template Matching and Viola-Jones work without downloading deep-learning models.

DeepFace and FaceNet download/load their required models when first used. If the environment cannot provide them, the application stays open and explains the limitation instead of generating fake results.

## How Each Algorithm Works
### Template Matching
`cv2.matchTemplate()` slides a template over a larger image and `cv2.minMaxLoc()` selects the best match.

### Viola-Jones
OpenCV's Haar Cascade uses Haar-like features, integral images, AdaBoost and a cascade classifier to detect faces.

### DeepFace
DeepFace provides deep-learning-based facial analysis and verification.

### FaceNet
The FaceNet-compatible `Facenet512` model represents a face as an embedding vector. VisionLab compares embeddings using cosine similarity and Euclidean distance.

## Future Improvements
- Additional real-world datasets
- More detector/model choices
- Exportable experiment reports
- Benchmark comparisons across algorithms
