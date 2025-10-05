import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

def compute_embedding_bgr(frame_bgr) -> np.ndarray:
    reps = DeepFace.represent(
        img_path=frame_bgr,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    if isinstance(reps, list) and reps and "embedding" in reps[0]:
        return np.array(reps[0]["embedding"], dtype=np.float32)
    raise ValueError("No face/embedding")