import cv2
import numpy as np

class BaseFilter:
    def __init__(self, name="Base Filter"):
        """
        Inicializa el filtro base.
        :param name: Nombre identificativo del filtro.
        """
        self.name = name

    def apply(self, frame, face_landmarks):
        """
        Aplica el filtro al frame actual utilizando los puntos de referencia faciales (landmarks).
        Debe ser sobreescrito por las subclases.
        
        :param frame: Imagen original (BGR numpy array).
        :param face_landmarks: Lista de puntos faciales o None si no hay rostros.
        :return: El frame modificado con el filtro aplicado.
        """
        # Por defecto, el filtro base no realiza cambios y retorna el frame original.
        return frame
