import os

# Reduce unnecessary TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Check FaceNet / DeepFace availability
# ---------------------------------------------------------
def facenet_available():
    try:
        from deepface import DeepFace
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# Load image safely
# ---------------------------------------------------------
def _load_image(image_source):

    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    if isinstance(image_source, np.ndarray):
        return Image.fromarray(image_source).convert("RGB")

    if hasattr(image_source, "read"):
        return Image.open(image_source).convert("RGB")

    return Image.open(image_source).convert("RGB")


# ---------------------------------------------------------
# Resize large images before FaceNet processing
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Get FaceNet embedding
# ---------------------------------------------------------
def _get_embedding(image):

    try:
        from deepface import DeepFace

        # Resize before sending image to the model
        pil_image = _resize_for_ai(image, max_size=1024)

        image_array = np.asarray(
            pil_image,
            dtype=np.uint8
        )

        result = DeepFace.represent(
            img_path=image_array,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if not result:
            return None, "No face embedding was returned."

        if isinstance(result, list):
            result = result[0]

        embedding = result.get("embedding")

        if embedding is None:
            return None, "FaceNet did not return an embedding."

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        if embedding.size == 0:
            return None, "The returned embedding is empty."

        return embedding, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Generate embedding
# ---------------------------------------------------------
def generate_embedding(image):

    embedding, error = _get_embedding(image)

    if embedding is None:
        return {
            "ok": False,
            "message": (
                "FaceNet model is not currently available.\n\n"
                f"Technical error: {error}\n\n"
                "Try:\n"
                "pip install -U deepface tf-keras"
            )
        }

    # Keep only first 32 dimensions for compact UI visualization
    preview = {
        f"Dimension {i + 1}": float(value)
        for i, value in enumerate(embedding[:32])
    }

    return {
        "ok": True,
        "dimension": int(len(embedding)),
        "embedding": embedding,
        "preview": preview
    }


# ---------------------------------------------------------
# Compare two FaceNet embeddings
# ---------------------------------------------------------
def compare_embeddings(image1, image2):

    # Generate first embedding
    embedding1, error1 = _get_embedding(image1)

    if embedding1 is None:
        return {
            "ok": False,
            "message": f"First image embedding failed: {error1}"
        }

    # Generate second embedding
    embedding2, error2 = _get_embedding(image2)

    if embedding2 is None:
        return {
            "ok": False,
            "message": f"Second image embedding failed: {error2}"
        }

    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    # Make sure dimensions match
    if embedding1.shape != embedding2.shape:
        return {
            "ok": False,
            "message": (
                f"Embedding dimensions do not match: "
                f"{embedding1.shape} vs {embedding2.shape}"
            )
        }

    # -----------------------------------------------------
    # Euclidean distance
    # -----------------------------------------------------
    distance = float(
        np.linalg.norm(
            embedding1 - embedding2
        )
    )

    # -----------------------------------------------------
    # Cosine similarity
    # -----------------------------------------------------
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:

        similarity = 0.0

    else:

        similarity = float(
            np.dot(
                embedding1,
                embedding2
            ) / (norm1 * norm2)
        )

    # Keep similarity within valid range
    similarity = max(
        -1.0,
        min(1.0, similarity)
    )

    # Demo threshold
    match = similarity >= 0.75

    return {
        "ok": True,
        "similarity": similarity,
        "distance": distance,
        "match": bool(match)
    }