import os

# Reduce TensorFlow resource usage on Streamlit Cloud.
# These must be set before TensorFlow is imported.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
from PIL import Image


def deepface_available():
    """
    Check whether DeepFace can be imported successfully.
    """
    try:
        from deepface import DeepFace
        return True
    except Exception:
        return False


def _load_image(image_source):
    """
    Convert different image inputs into a PIL RGB image.
    """

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source).convert("RGB")

    if hasattr(image_source, "read"):
        return Image.open(image_source).convert("RGB")

    return Image.open(image_source).convert("RGB")


def _get_deepface():
    """
    Safely import DeepFace.
    """

    try:
        from deepface import DeepFace
        return DeepFace, None

    except Exception as e:
        return None, str(e)


def _clean_probability_dictionary(data):
    """
    Convert NumPy values into normal Python floats.

    Example:
        np.float32(88.50042)

    becomes:
        88.5
    """

    if not isinstance(data, dict):
        return data

    cleaned = {}

    for key, value in data.items():

        try:
            cleaned[key] = round(float(value), 2)

        except (TypeError, ValueError):
            cleaned[key] = value

    return cleaned


def _clean_deepface_result(result):
    """
    Clean DeepFace output.
    """

    if isinstance(result, list):

        cleaned_results = []

        for item in result:

            if isinstance(item, dict):

                item = item.copy()

                if "gender" in item:
                    item["gender"] = _clean_probability_dictionary(
                        item["gender"]
                    )

                if "emotion" in item:
                    item["emotion"] = _clean_probability_dictionary(
                        item["emotion"]
                    )

                if "race" in item:
                    item["race"] = _clean_probability_dictionary(
                        item["race"]
                    )

            cleaned_results.append(item)

        return cleaned_results

    if isinstance(result, dict):

        result = result.copy()

        if "gender" in result:
            result["gender"] = _clean_probability_dictionary(
                result["gender"]
            )

        if "emotion" in result:
            result["emotion"] = _clean_probability_dictionary(
                result["emotion"]
            )

        if "race" in result:
            result["race"] = _clean_probability_dictionary(
                result["race"]
            )

        return result

    return result


def _release_attribute_model():
    """
    Release cached DeepFace demographic models.

    DeepFace keeps Age, Gender, Emotion and Race models
    inside a global model cache. Clearing the facial
    attribute cache prevents all four large models from
    staying in memory at the same time.
    """

    try:

        from deepface.modules import modeling

        if hasattr(modeling, "cached_models"):

            if "facial_attribute" in modeling.cached_models:

                modeling.cached_models["facial_attribute"].clear()

    except Exception:
        pass

    # Ask TensorFlow/Keras to release unused memory.
    try:

        import tensorflow as tf

        tf.keras.backend.clear_session()

    except Exception:
        pass

    gc.collect()


def _run_single_attribute(
    DeepFace,
    image_array,
    action
):
    """
    Run one DeepFace attribute at a time.

    This is intentionally done one-by-one to reduce
    peak RAM usage on Streamlit Cloud.
    """

    try:

        result = DeepFace.analyze(
            img_path=image_array,
            actions=[action],
            enforce_detection=False,
            detector_backend="opencv"
        )

        result = _clean_deepface_result(result)

        if isinstance(result, list):

            if len(result) == 0:
                return None

            return result[0]

        if isinstance(result, dict):
            return result

        return None

    finally:

        # Release the model before loading the next one.
        _release_attribute_model()


def analyze_face(image_source):
    """
    Analyze a face using DeepFace.

    Age, gender, emotion and race are analyzed
    one at a time to reduce memory usage.

    Returns:
        age
        gender probabilities
        emotion probabilities
        race probabilities
        dominant values
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

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = _load_image(image_source)

        image_array = np.array(image)

        # -------------------------------------------------
        # AGE
        # -------------------------------------------------

        age_result = _run_single_attribute(
            DeepFace,
            image_array,
            "age"
        )

        if age_result is None:
            return {
                "ok": False,
                "message": "DeepFace could not analyze the face."
            }

        age = age_result.get("age", "N/A")

        # -------------------------------------------------
        # GENDER
        # -------------------------------------------------

        gender_result = _run_single_attribute(
            DeepFace,
            image_array,
            "gender"
        )

        gender_data = {}

        if gender_result:

            gender_data = gender_result.get(
                "gender",
                {}
            )

        gender_data = _clean_probability_dictionary(
            gender_data
        )

        if isinstance(gender_data, dict) and gender_data:

            dominant_gender = max(
                gender_data,
                key=gender_data.get
            )

        else:

            dominant_gender = "N/A"

        # -------------------------------------------------
        # EMOTION
        # -------------------------------------------------

        emotion_result = _run_single_attribute(
            DeepFace,
            image_array,
            "emotion"
        )

        emotion_data = {}

        if emotion_result:

            emotion_data = emotion_result.get(
                "emotion",
                {}
            )

        emotion_data = _clean_probability_dictionary(
            emotion_data
        )

        if isinstance(emotion_data, dict) and emotion_data:

            dominant_emotion = max(
                emotion_data,
                key=emotion_data.get
            )

        else:

            dominant_emotion = "N/A"

        # -------------------------------------------------
        # RACE
        # -------------------------------------------------

        race_result = _run_single_attribute(
            DeepFace,
            image_array,
            "race"
        )

        race_data = {}

        if race_result:

            race_data = race_result.get(
                "race",
                {}
            )

        race_data = _clean_probability_dictionary(
            race_data
        )

        if isinstance(race_data, dict) and race_data:

            dominant_race = max(
                race_data,
                key=race_data.get
            )

        else:

            dominant_race = "N/A"

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        combined_result = {
            "age": age,
            "gender": gender_data,
            "dominant_gender": dominant_gender,
            "emotion": emotion_data,
            "dominant_emotion": dominant_emotion,
            "race": race_data,
            "dominant_race": dominant_race
        }

        return {
            "ok": True,

            "image": image,

            "result": combined_result,

            "age": age,

            "gender": gender_data,

            "dominant_gender": dominant_gender,

            "emotion": emotion_data,

            "dominant_emotion": dominant_emotion,

            "race": race_data,

            "dominant_race": dominant_race
        }

    except Exception as e:

        return {
            "ok": False,
            "message": f"DeepFace analysis failed: {e}"
        }


def verify_faces(image1_source, image2_source):
    """
    Compare two faces using DeepFace.

    Facenet is explicitly selected because it is
    lighter than the default VGG-Face verification
    model and is suitable for the cloud deployment.
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

        # Load images
        image1 = _load_image(image1_source)
        image2 = _load_image(image2_source)

        img1 = np.array(image1)
        img2 = np.array(image2)

        # -------------------------------------------------
        # FACE VERIFICATION
        # -------------------------------------------------

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )

        # Clean NumPy values
        if isinstance(result, dict):

            result = result.copy()

            for key, value in result.items():

                if isinstance(value, np.generic):

                    result[key] = value.item()

        verified = result.get(
            "verified",
            False
        )

        distance = result.get(
            "distance",
            None
        )

        threshold = result.get(
            "threshold",
            None
        )

        if isinstance(distance, np.generic):
            distance = float(distance)

        if isinstance(threshold, np.generic):
            threshold = float(threshold)

        return {
            "ok": True,

            "result": result,

            "verified": bool(verified),

            "distance": distance,

            "threshold": threshold,

            "image1": image1,

            "image2": image2
        }

    except Exception as e:

        return {
            "ok": False,
            "message": f"DeepFace verification failed: {e}"
        }

    finally:

        _release_attribute_model()