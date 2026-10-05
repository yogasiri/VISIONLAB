import numpy as np
from PIL import Image


def facenet_available():
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


def _get_embedding(image):

    try:
        from deepface import DeepFace

        pil_image = _load_image(image)
        image_array = np.array(pil_image)

        result = DeepFace.represent(
            img_path=image_array,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if isinstance(result, list):
            if len(result) == 0:
                return None, "No face embedding was returned."
            result = result[0]

        embedding = result.get("embedding")

        if embedding is None:
            return None, "FaceNet did not return an embedding."

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        return embedding, None

    except Exception as e:
        return None, str(e)


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

    # First 32 values only for compact visualization
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


def compare_embeddings(image1, image2):

    embedding1, error1 = _get_embedding(image1)

    if embedding1 is None:
        return {
            "ok": False,
            "message": f"First image embedding failed: {error1}"
        }

    embedding2, error2 = _get_embedding(image2)

    if embedding2 is None:
        return {
            "ok": False,
            "message": f"Second image embedding failed: {error2}"
        }

    # Make sure both are vectors
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    if embedding1.shape != embedding2.shape:
        return {
            "ok": False,
            "message": (
                f"Embedding dimensions do not match: "
                f"{embedding1.shape} vs {embedding2.shape}"
            )
        }

    # Euclidean distance
    distance = float(
        np.linalg.norm(
            embedding1 - embedding2
        )
    )

    # Cosine similarity
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

    # FaceNet-compatible similarity display
    similarity = max(-1.0, min(1.0, similarity))

    # Demo threshold
    match = similarity >= 0.75

    return {
        "ok": True,
        "similarity": similarity,
        "distance": distance,
        "match": bool(match)
    }