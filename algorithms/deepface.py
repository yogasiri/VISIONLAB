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
    Convert NumPy float values returned by DeepFace
    into normal Python floats rounded to 2 decimals.

    Example:

    np.float32(88.50042)

    becomes:

    88.50
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
    Clean DeepFace's output so that NumPy values such as
    np.float32(...) are removed.

    Gender, emotion and race probabilities are converted
    into normal Python floats.
    """

    # DeepFace normally returns a list
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

    # Safety handling if DeepFace returns a dictionary
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


def analyze_face(image_source):
    """
    Analyze a face using DeepFace.

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

        # Convert input to PIL image
        image = _load_image(image_source)

        # Convert PIL image to NumPy array
        image_array = np.array(image)

        # Run DeepFace analysis
        result = DeepFace.analyze(
            img_path=image_array,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        # Clean NumPy float values
        result = _clean_deepface_result(result)

        # DeepFace normally returns a list
        if isinstance(result, list):

            if len(result) == 0:
                return {
                    "ok": False,
                    "message": "DeepFace returned no analysis result."
                }

            result = result[0]

        if not isinstance(result, dict):
            return {
                "ok": False,
                "message": "DeepFace returned an unexpected result format."
            }

        # -------------------------------------------------
        # AGE
        # -------------------------------------------------

        age = result.get("age", "N/A")

        # -------------------------------------------------
        # GENDER
        # -------------------------------------------------

        gender_data = result.get("gender", {})

        if isinstance(gender_data, dict) and gender_data:

            # Highest probability gender
            dominant_gender = max(
                gender_data,
                key=gender_data.get
            )

        else:

            dominant_gender = result.get(
                "dominant_gender",
                "N/A"
            )

        # -------------------------------------------------
        # EMOTION
        # -------------------------------------------------

        emotion_data = result.get("emotion", {})

        if isinstance(emotion_data, dict) and emotion_data:

            dominant_emotion = max(
                emotion_data,
                key=emotion_data.get
            )

        else:

            dominant_emotion = result.get(
                "dominant_emotion",
                "N/A"
            )

        # -------------------------------------------------
        # RACE
        # -------------------------------------------------

        race_data = result.get("race", {})

        if isinstance(race_data, dict) and race_data:

            dominant_race = max(
                race_data,
                key=race_data.get
            )

        else:

            dominant_race = result.get(
                "dominant_race",
                "N/A"
            )

        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return {
            "ok": True,

            "image": image,

            "result": result,

            "age": age,

            # Full probability dictionary
            "gender": gender_data,

            # Highest probability gender
            "dominant_gender": dominant_gender,

            # Full emotion probabilities
            "emotion": emotion_data,

            # Highest probability emotion
            "dominant_emotion": dominant_emotion,

            # Full race probabilities
            "race": race_data,

            # Highest probability race
            "dominant_race": dominant_race
        }

    except Exception as e:

        return {
            "ok": False,
            "message": f"DeepFace analysis failed: {e}"
        }


def verify_faces(image1_source, image2_source):
    """
    Compare two faces using DeepFace verification.
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

        # Load both images
        image1 = _load_image(image1_source)
        image2 = _load_image(image2_source)

        # Convert to NumPy arrays
        img1 = np.array(image1)
        img2 = np.array(image2)

        # Verify faces
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            enforce_detection=False,
            detector_backend="opencv"
        )

        # Clean NumPy values if present
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

        # Convert NumPy numbers if necessary
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