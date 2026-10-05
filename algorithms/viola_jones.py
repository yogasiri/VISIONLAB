import cv2
import numpy as np
from PIL import Image
import time


def detect_faces(image_source):
    """
    Viola-Jones face detection using OpenCV Haar Cascade.

    Returns the keys expected by VISIONLAB app.py:
    - ok
    - image
    - face_count
    - time_ms
    - resolution
    - faces
    """

    try:

        # -----------------------------------
        # 1. Load image
        # -----------------------------------

        if isinstance(image_source, np.ndarray):

            image = image_source.copy()

            if image.ndim == 2:
                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB
                )

        elif hasattr(image_source, "convert"):

            image = np.array(
                image_source.convert("RGB")
            )

        elif hasattr(image_source, "read"):

            image = np.array(
                Image.open(image_source).convert("RGB")
            )

        else:

            image = np.array(
                Image.open(image_source).convert("RGB")
            )

        # -----------------------------------
        # 2. Convert RGB to BGR
        # -----------------------------------

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        # -----------------------------------
        # 3. Convert to grayscale
        # -----------------------------------

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        # -----------------------------------
        # 4. Load Haar Cascade
        # -----------------------------------

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if face_cascade.empty():

            return {
                "ok": False,
                "message": (
                    "Haar Cascade classifier "
                    "could not be loaded."
                )
            }

        # -----------------------------------
        # 5. Start timer
        # -----------------------------------

        start_time = time.perf_counter()

        # -----------------------------------
        # 6. Detect faces
        # -----------------------------------

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # -----------------------------------
        # 7. Calculate processing time
        # -----------------------------------

        processing_time = (
            time.perf_counter() - start_time
        )

        # Convert seconds → milliseconds

        time_ms = processing_time * 1000

        # -----------------------------------
        # 8. Handle no faces
        # -----------------------------------

        if faces is None:
            faces = []

        # -----------------------------------
        # 9. Draw bounding boxes
        # -----------------------------------

        output = image.copy()

        face_count = len(faces)

        for index, face in enumerate(
            faces,
            start=1
        ):

            # OpenCV format:
            # x, y, width, height

            x, y, w, h = [
                int(value)
                for value in face
            ]

            # --------------------------------
            # Bounding box
            # --------------------------------

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (66, 224, 255),
                3
            )

            # --------------------------------
            # Face label
            # --------------------------------

            cv2.putText(
                output,
                f"Face {index}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (66, 224, 255),
                2,
                cv2.LINE_AA
            )

        # -----------------------------------
        # 10. Resolution
        # -----------------------------------

        resolution = (
            f"{image.shape[1]} x "
            f"{image.shape[0]}"
        )

        # -----------------------------------
        # 11. Return result
        # -----------------------------------

        return {

            "ok": True,

            # Processed image
            "image": Image.fromarray(output),

            # Number of detected faces
            "face_count": face_count,

            # Processing time in milliseconds
            "time_ms": float(time_ms),

            # Keep this too for compatibility
            "processing_time": float(
                processing_time
            ),

            # Image resolution
            "resolution": resolution,

            # Face coordinates
            "faces": faces
        }

    except Exception as e:

        return {
            "ok": False,
            "message": (
                f"Face detection could not run: {e}"
            )
        }


# ------------------------------------------------
# Compatibility function
# ------------------------------------------------

def run_viola_jones(image_source):
    return detect_faces(image_source)