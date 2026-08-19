import cv2
import numpy as np
from .base_filter import BaseFilter

class FiltroNariz(BaseFilter):
    def __init__(self):
        super().__init__(name="Filtro de Nariz Roja")

    def apply(self, frame, face_landmarks):
        """
        Dibuja un círculo rojo en la punta de la nariz.
        """
        # 1. Verificamos que tengamos puntos detectados
        if not face_landmarks:
            return frame

        # 2. Obtenemos las dimensiones de la imagen
        alto, ancho, _ = frame.shape

        # 3. El punto 1 es la punta de la nariz en la malla de MediaPipe
        punto_nariz = face_landmarks[1]

        # 4. Convertimos las coordenadas normalizadas (0.0 a 1.0) a píxeles reales
        cx = int(punto_nariz.x * ancho)
        cy = int(punto_nariz.y * alto)

        # 5. Dibujamos un efecto (por ejemplo, una nariz de payaso)
        # cv2.circle(imagen, centro, radio, color, grosor)
        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)

        return frame
