import cv2
import numpy as np
from .base_filter import BaseFilter

class MustacheFilter(BaseFilter):
    NOSE_TIP = 1  # Índice del landmark de la punta de la nariz

    def __init__(self, facedetector, png_path, offset_x, offset_y, target):
        super().__init__()
        
        self.facedetector = facedetector
        self.png_path = str(png_path)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.target = target
        self.mustache_image = cv2.imread(self.png_path, cv2.IMREAD_UNCHANGED)

    def landmark_to_xy(self, image, landmark):
        return self.facedetector.landmark_to_coordinates(image, landmark)

    def apply(self, frame, face_landmarks):
        if not face_landmarks:
            return frame 

        nose = face_landmarks[self.NOSE_TIP]
        x_nose, y_nose = self.landmark_to_xy(frame, nose)
        
        oh, ow = self.mustache_image.shape[:2]
        scale = self.target / ow

        new_h = max(int(oh * scale), 1)
        new_w = max(int(ow * scale), 1)

        resized_mustache = cv2.resize(self.mustache_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        x = x_nose + self.offset_x - new_w // 2
        y = y_nose + self.offset_y - new_h // 2

        mustache_rgb = resized_mustache[:, :, :3]
        mustache_alpha = resized_mustache[:, :, 3] / 255.0

        y1, y2 = max(0, y), min(frame.shape[0], y + new_h)
        x1, x2 = max(0, x), min(frame.shape[1], x + new_w)
        
        y1o, y2o = max(0, -y), min(new_h, new_h - (y + new_h - frame.shape[0]))
        x1o, x2o = max(0, -x), min(new_w, new_w - (x + new_w - frame.shape[1]))

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return frame

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (
                mustache_alpha[y1o:y2o, x1o:x2o] * mustache_rgb[y1o:y2o, x1o:x2o, c] +
                (1.0 - mustache_alpha[y1o:y2o, x1o:x2o]) * frame[y1:y2, x1:x2, c]
            )

        return frame