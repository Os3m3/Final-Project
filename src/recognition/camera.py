import cv2
from typing import Tuple, Optional

class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {index}")

    def read(self) -> Tuple[bool, Optional[any]]:
        return self.cap.read()

    def release(self):
        if getattr(self, "cap", None):
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()