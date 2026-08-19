import cv2
from .base_filter import BaseFilter

class FiltroNariz(BaseFilter):
    def __init__(self, detector):
        """
        Inicializa el filtro de nariz.
        :param detector: Instancia de FaceDetection para convertir coordenadas.
        """
        # CORRECCIÓN: Llamamos a super() sin el argumento 'name'
        super().__init__() 
        self.name = "Filtro Nariz" # Asignamos el nombre directamente
        self.detector = detector

    def apply(self, frame, face_landmarks):
        """
        Aplica un círculo rojo en la punta de la nariz.
        """
        if face_landmarks:
            nose_landmark = face_landmarks[1]
            x, y = self.detector.landmark_to_coordinates(frame, nose_landmark)
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            
        return frame
