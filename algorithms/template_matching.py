import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def _load_image(image_source):
    """Load an image from PIL, NumPy array, uploaded file, or file path."""

    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")

    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return Image.fromarray(image_source).convert("RGB")

        if image_source.ndim == 3:
            return Image.fromarray(image_source).convert("RGB")

    if hasattr(image_source, "convert"):
        return image_source.convert("RGB")

    if hasattr(image_source, "read"):
        return Image.open(image_source).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image_source)}")


def run_template_matching(main_image, template_image):
    try:
        # Load both images safely
        main_image = _load_image(main_image)
        template_image = _load_image(template_image)

        # PIL -> NumPy
        main = np.array(main_image)
        template = np.array(template_image)

        # RGB -> Grayscale
        main_gray = cv2.cvtColor(main, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

        mh, mw = main_gray.shape
        th, tw = template_gray.shape

        # Template must be smaller
        if th >= mh or tw >= mw:
            return {
                "ok": False,
                "message": "Template image must be smaller than the main image."
            }

        # OpenCV Template Matching
        result = cv2.matchTemplate(
            main_gray,
            template_gray,
            cv2.TM_CCOEFF_NORMED
        )

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        x, y = max_loc

        # Draw bounding rectangle
        output = main.copy()

        cv2.rectangle(
            output,
            (x, y),
            (x + tw, y + th),
            (66, 224, 255),
            4
        )

        # Add similarity text
        cv2.putText(
            output,
            f"Best Match: {max_val * 100:.2f}%",
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (66, 224, 255),
            2,
            cv2.LINE_AA
        )

        return {
            "ok": True,
            "image": Image.fromarray(output),
            "similarity": float(max_val),
            "location": (int(x), int(y)),
            "template_size": (int(tw), int(th))
        }

    except Exception as e:
        return {
            "ok": False,
            "message": f"Template matching could not run: {e}"
        }