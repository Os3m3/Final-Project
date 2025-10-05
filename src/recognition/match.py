import pickle
import numpy as np
from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from src.db.models import User
from src.recognition.embeddings import compute_embedding_from_np

def load_gallery(session: Session) -> List[Tuple[int, str, np.ndarray]]:
    gallery = []
    for u in session.query(User).all():
        if u.face_embedding:
            emb = pickle.loads(u.face_embedding).astype(np.float32)
            n = np.linalg.norm(emb)
            if n > 0: emb = emb / n
            gallery.append((u.user_id, u.name, emb))
    return gallery

def best_match(img_np: np.ndarray, gallery: List[Tuple[int, str, np.ndarray]], threshold: float = 0.70) -> Optional[Tuple[int, str, float]]:
    if not gallery:
        return None
    q = compute_embedding_from_np(img_np).astype(np.float32)
    nq = np.linalg.norm(q)
    if nq > 0: q = q / nq
    scores = [(uid, name, float(np.dot(q, emb))) for uid, name, emb in gallery]  # cosine via dot (normed)
    uid, name, sc = max(scores, key=lambda x: x[2])
    return (uid, name, sc) if sc >= threshold else None