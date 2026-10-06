import os

# ---------------------------------------------------------
# REDUCE TENSORFLOW RESOURCE USAGE
# These must be set before TensorFlow is imported.
# ---------------------------------------------------------

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# FIXED DEMO VALUES
# ---------------------------------------------------------
# These are NOT predicted by DeepFace.
# They are displayed as fixed educational demo values.

FIXED_AGE = "21"
FIXED_RACE = "Asian"


# ---------------------------------------------------------
# CHECK DEEPFACE
# ---------------------------------------------------------

def deepface_available():
    """
    Check whether DeepFace can be imported successfully.
    """

    try:
        from deepface import DeepFace
        return True

    except Exception:
        return False


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RESIZE IMAGE FOR AI PROCESSING
# ---------------------------------------------------------

def _resize_for_ai(image, max_size=1024):
    """
    Resize large images before sending them to AI models.

    This reduces unnecessary CPU and RAM usage while
    keeping the image aspect ratio.
    """

    image = image.copy()

    image.thumbnail((max_size, max_size))

    return image


# ---------------------------------------------------------
# GET DEEPFACE
# ---------------------------------------------------------

def _get_deepface():
    """
    Safely import DeepFace only when it is required.
    """

    try:

        from deepface import DeepFace

        return DeepFace, None

    except Exception as e:

        return None, str(e)


# ---------------------------------------------------------
# CLEAN PROBABILITY DICTIONARY
# ---------------------------------------------------------

def _clean_probability_dictionary(data):
    """
    Convert NumPy probability values into normal Python floats.

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


# ---------------------------------------------------------
# CLEAN DEEPFACE RESULT
# ---------------------------------------------------------

def _clean_deepface_result(result):
    """
    Clean DeepFace output so NumPy values do not appear
    in the Streamlit interface.
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

        return result

    return result


# ---------------------------------------------------------
# RELEASE DEEPFACE ATTRIBUTE MODEL
# ---------------------------------------------------------

def _release_attribute_model():
    """
    Release cached DeepFace facial attribute models.

    This prevents Gender and Emotion models from staying
    unnecessarily in memory at the same time.
    """

    try:

        from deepface.modules import modeling

        if hasattr(modeling, "cached_models"):

            if "facial_attribute" in modeling.cached_models:

                modeling.cached_models["facial_attribute"].clear()

    except Exception:

        pass

    # Release TensorFlow/Keras resources.

    try:

        import tensorflow as tf

        tf.keras.backend.clear_session()

    except Exception:

        pass

    gc.collect()


# ---------------------------------------------------------
# RUN ONE DEEPFACE ATTRIBUTE
# ---------------------------------------------------------

def _run_single_attribute(
    DeepFace,
    image_array,
    action
):
    """
    Run one DeepFace attribute at a time.

    Only Gender and Emotion use actual DeepFace models.
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

        # Release the model before running the next attribute.

        _release_attribute_model()


# =========================================================
# DEEPFACE FACE ANALYSIS
# =========================================================

def analyze_face(image_source):
    """
    Analyze a face using DeepFace.

    REAL AI ANALYSIS:
        - Gender
        - Emotion

    FIXED DEMO VALUES:
        - Age = 21
        - Race = Asian

    Age and Race models are NOT loaded.

    Face Verification is intentionally NOT included here.
    Face similarity/embedding is handled separately by FaceNet.
    """

    # -----------------------------------------------------
    # LOAD DEEPFACE
    # -----------------------------------------------------

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

        # Resize large images before AI processing.

        image = _resize_for_ai(image)

        image_array = np.array(image)

        # =================================================
        # FIXED AGE
        # =================================================

        # No Age model is loaded.

        age = FIXED_AGE

        # =================================================
        # GENDER
        # =================================================

        # REAL DeepFace Gender model.

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

        # =================================================
        # EMOTION
        # =================================================

        # REAL DeepFace Emotion model.

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

        # =================================================
        # FIXED RACE
        # =================================================

        # No Race model is loaded.

        race = FIXED_RACE

        dominant_race = FIXED_RACE

        # =================================================
        # FINAL RESULT
        # =================================================

        combined_result = {

            "age": age,

            "gender": gender_data,

            "dominant_gender": dominant_gender,

            "emotion": emotion_data,

            "dominant_emotion": dominant_emotion,

            "race": race,

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

            "race": race,

            "dominant_race": dominant_race

        }

    except Exception as e:

        return {

            "ok": False,

            "message": f"DeepFace analysis failed: {e}"

        }