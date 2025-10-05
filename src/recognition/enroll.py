import argparse, pickle, cv2, numpy as np
from typing import List
from src.db.base import get_session
from src.db.models import User
from src.recognition.camera import Camera
from src.recognition.embeddings import compute_embedding_bgr

HINT = "SPACE: capture  |  ESC: quit"

def _put(frame, text, y=28, color=(0,255,0)):
    cv2.putText(frame, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def enroll(name: str, samples: int = 5, cam_index: int = 0):
    cam = Camera(index=cam_index)
    embs: List[np.ndarray] = []
    try:
        while len(embs) < samples:
            ok, frame = cam.read()
            if not ok: continue
            _put(frame, f"Enroll: {name}  [{len(embs)}/{samples}]  {HINT}")
            cv2.imshow("Enroll", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            if k == 32:
                try:
                    embs.append(compute_embedding_bgr(frame))
                    _put(frame, f"Captured {len(embs)}/{samples}", y=60)
                    cv2.imshow("Enroll", frame); cv2.waitKey(400)
                except Exception:
                    _put(frame, "Face not detected, try again", y=60, color=(0,0,255))
                    cv2.imshow("Enroll", frame); cv2.waitKey(700)
        cv2.destroyWindow("Enroll")
    finally:
        cam.release(); cv2.destroyAllWindows()

    if not embs: return
    avg_emb = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
    emb_bytes = pickle.dumps(avg_emb, protocol=pickle.HIGHEST_PROTOCOL)

    with get_session() as s:
        u = s.query(User).filter(User.name == name).one_or_none()
        if u is None:
            s.add(User(name=name, face_embedding=emb_bytes))
        else:
            u.face_embedding = emb_bytes
        s.commit()
    print(f"[OK] Enrollment saved for {name}")

def list_users():
    with get_session() as s:
        for u in s.query(User).order_by(User.user_id).all():
            print(f"{u.user_id}: {u.name} | embedding: {'yes' if u.face_embedding else 'no'}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("enroll"); p1.add_argument("name"); p1.add_argument("--samples", type=int, default=5); p1.add_argument("--cam", type=int, default=0)
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "enroll": enroll(args.name, args.samples, args.cam)
    if args.cmd == "list":   list_users()

if __name__ == "__main__": main()