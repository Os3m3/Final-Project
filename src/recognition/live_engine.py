import time, pickle, cv2, numpy as np, argparse
from typing import List, Tuple, Optional
from src.db.base import get_session
from src.db.models import User
from src.recognition.camera import Camera
from src.recognition.embeddings import compute_embedding_bgr
from src.attendance.gate_logic import now_local, active_gate, upsert_attendance

THRESHOLD = 0.75
STABLE_FRAMES = 5
COOLDOWN_SEC = 8
RELOAD_GALLERY_SEC = 15

def _put(frame, text, y=28, color=(0,255,0)):
    cv2.putText(frame, text, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def _load_users(verbose=False) -> List[Tuple[int,str,np.ndarray]]:
    from numpy import float32
    users=[]
    with get_session() as s:
        for u in s.query(User).all():
            if u.face_embedding:
                emb = pickle.loads(u.face_embedding).astype(float32)
                n = np.linalg.norm(emb)
                if n>0: emb = emb/n
                users.append((u.user_id, u.name, emb))
    if verbose: print(f"[live] loaded {len(users)} users with embeddings")
    return users

def _best_match(frame_bgr, gallery) -> Optional[Tuple[int,str,float]]:
    try:
        q = compute_embedding_bgr(frame_bgr).astype(np.float32)
    except Exception:
        return None
    nq = np.linalg.norm(q);  q = q/nq if nq>0 else q
    if not gallery: return None
    uid,name,sc = max(((uid,name,float(np.dot(q,emb))) for uid,name,emb in gallery), key=lambda x:x[2])
    return (uid,name,sc) if sc>=THRESHOLD else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.verbose: print(f"[live] starting with cam index {args.cam}")

    cam = Camera(index=args.cam)
    ok, _ = cam.read()
    if not ok:
        print("[live] ERROR: cannot read from camera. Try a different --cam index.")
        # show a small window so you know it ran
        canvas = np.zeros((300,600,3), dtype=np.uint8)
        _put(canvas, "Cannot open camera", y=80, color=(0,0,255))
        _put(canvas, "Try: --cam 1  or  --cam 2", y=120, color=(0,255,255))
        cv2.imshow("Live Engine", canvas); cv2.waitKey(3000)
        cam.release(); cv2.destroyAllWindows(); return

    gallery = _load_users(verbose=args.verbose)
    last_reload = time.time()

    last_written = {}
    stable_id = None; stable_count = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                frame = np.zeros((300,600,3), dtype=np.uint8)
                _put(frame, "Camera read failed", y=80, color=(0,0,255))

            now = now_local(); gate = active_gate(now)
            _put(frame, "ESC: quit", y=24, color=(0,255,255))
            _put(frame, f"{now.strftime('%H:%M:%S')}  |  Gate: {gate if gate else '-'}", y=48, color=(255,255,0))

            if not gallery:
                _put(frame, "No enrolled users. Add a user in the app.", y=80, color=(0,0,255))
                cv2.imshow("Live Engine", frame)
            else:
                match = _best_match(frame, gallery)
                if match is None:
                    stable_id, stable_count = None, 0
                    _put(frame, "Unknown / no face", y=80, color=(0,0,255))
                    cv2.imshow("Live Engine", frame)
                else:
                    uid, name, score = match
                    _put(frame, f"{name} ({score:.2f})", y=80, color=(0,255,0))
                    if stable_id == uid: stable_count += 1
                    else: stable_id, stable_count = uid, 1

                    if stable_count >= STABLE_FRAMES:
                        if gate is None:
                            _put(frame, "Outside gate window", y=112, color=(0,165,255))
                        elif uid in last_written and (time.time()-last_written[uid])<COOLDOWN_SEC:
                            _put(frame, "Already recorded (cooldown)", y=112, color=(0,165,255))
                        else:
                            try:
                                with get_session() as s:
                                    action = upsert_attendance(s, uid, gate, now)
                                last_written[uid] = time.time()
                                if action == "checkout":
                                    _put(frame, f"✅ Checked out {name}", y=112, color=(0,255,255))
                                elif action and action.startswith("checkin"):
                                    if gate == 4:
                                        _put(frame, "Evening check-in (scan again to checkout)", y=112, color=(0,255,255))
                                    else:
                                        _put(frame, f"✅ Check-in recorded ({name}) [Gate {action[-1]}]", y=112, color=(0,255,0))
                                elif action == "noop":
                                    _put(frame, "No change (already set)", y=112, color=(0,165,255))
                                else:
                                    _put(frame, "Recorded", y=112, color=(0,255,255))
                                if args.verbose: print(f"[live] action={action}, uid={uid}, gate={gate}")
                            except Exception:
                                _put(frame, "Write error – see console", y=112, color=(0,0,255))
                                import traceback; traceback.print_exc()
                        stable_count = 0
                    cv2.imshow("Live Engine", frame)

            if time.time() - last_reload >= RELOAD_GALLERY_SEC:
                gallery = _load_users(verbose=args.verbose)
                last_reload = time.time()

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cam.release()
        try: cv2.destroyAllWindows()
        except cv2.error: pass

if __name__ == "__main__":
    main()
