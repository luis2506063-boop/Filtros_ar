import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetection:
    def __init__(self, 
                 max_num_faces=1, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        # Necesitamos el archivo del modelo .task
        # Asumo que existe o se descargará.
        model_path = 'face_landmarker.task' 
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_faces(self, image):
        # Convertir a MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Procesar
        results = self.landmarker.detect(mp_image)
        
        return results.face_landmarks

    def draw_face(self, image):
        # API de Tasks no incluye 'drawing_utils' directamente como 'solutions'
        # Por ahora, solo retornamos la imagen original.
        return image
