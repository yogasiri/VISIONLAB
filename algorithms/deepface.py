import os

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image


FIXED_AGE = "25-35"
FIXED_RACE = "Asian"


def deepface_available():
    try:
        from deepface import DeepFace
        return True
    except Exception:
        return False


def _load_image(image_source):
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source).convert("RGB")

    if hasattr(image_source, "read"):
        return Image.open(image_source).convert("RGB")

    return Image.open(image_source).convert("RGB")


def _resize_for_ai(image, max_size=1024):
    image = _load_image(image)

    width, height = image.size

    if max(width, height) > max_size:
        scale = max_size / max(width, height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    return image


def _get_deepface():
    try:
        from deepface import DeepFace
        return DeepFace, None

    except Exception as e:
        return None, str(e)


def _clean_probability_dictionary(data):
    if not isinstance(data, dict):
        return data

    cleaned = {}

    for key, value in data.items():
        try:
            cleaned[key] = round(float(value), 2)
        except (TypeError, ValueError):
            cleaned[key] = value

    return cleaned


def analyze_face(image_source):
    """
    Perform DeepFace facial analysis.

    Actual models used:
    - Gender
    - Emotion

    Age and race are intentionally displayed as fixed educational
    demonstration values.

    Face verification is not included here.
    FaceNet handles face similarity/comparison separately.
    """

    DeepFace, error = _get_deepface()

    if DeepFace is None:
        return {
            "ok": False,
            "message": (
                "DeepFace could not be loaded.\n\n"
                f"Technical error: {error}\n\n"
                "Try running:\n"
                "pip install -U deepface tf-keras"
            )
        }

    try:
        image = _load_image(image_source)
        image = _resize_for_ai(image)

        image_array = np.asarray(image, dtype=np.uint8)

        # Run gender and emotion together.
        # DeepFace keeps the loaded models available for reuse.
        result = DeepFace.analyze(
            img_path=image_array,
            actions=["gender", "emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        if isinstance(result, list):
            if len(result) == 0:
                return {
                    "ok": False,
                    "message": "No facial analysis result was returned."
                }

            result = result[0]

        if not isinstance(result, dict):
            return {
                "ok": False,
                "message": "DeepFace returned an unexpected result."
            }

        gender_data = _clean_probability_dictionary(
            result.get("gender", {})
        )

        emotion_data = _clean_probability_dictionary(
            result.get("emotion", {})
        )

        if isinstance(gender_data, dict) and gender_data:
            dominant_gender = max(
                gender_data,
                key=gender_data.get
            )
        else:
            dominant_gender = "N/A"

        if isinstance(emotion_data, dict) and emotion_data:
            dominant_emotion = max(
                emotion_data,
                key=emotion_data.get
            )
        else:
            dominant_emotion = "N/A"

        combined_result = {
            "age": FIXED_AGE,
            "gender": gender_data,
            "dominant_gender": dominant_gender,
            "emotion": emotion_data,
            "dominant_emotion": dominant_emotion,
            "race": FIXED_RACE,
            "dominant_race": FIXED_RACE
        }

        return {
            "ok": True,
            "image": image,
            "result": combined_result,
            "age": FIXED_AGE,
            "gender": gender_data,
            "dominant_gender": dominant_gender,
            "emotion": emotion_data,
            "dominant_emotion": dominant_emotion,
            "race": FIXED_RACE,
            "dominant_race": FIXED_RACE
        }

    except Exception as e:
        return {
            "ok": False,
            "message": f"DeepFace analysis failed: {e}"
        }