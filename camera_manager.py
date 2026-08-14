import cv2

class CameraManager:
    def __init__(self, camera_index: int = 0):
        """
        Inicializa la cámara por defecto usando OpenCV.
        :param camera_index: Índice de la cámara a utilizar (por defecto 0).
        """
        self.camera_index = camera_index

        # Inicializa la captura de video desde la cámara seleccionada
        self.capture = cv2.VideoCapture(self.camera_index)

        # Verifica si la cámara se pudo abrir correctamente
        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {self.camera_index}")
    
    def get_frame(self):
        """
        Captura el frame actual de la cámara.
        :return: El frame (imagen) si la lectura fue exitosa, o None si hubo un error.
        """
        # Lee el siguiente frame del flujo de video
        # ret: Booleano que indica si la captura fue exitosa
        # frame: La imagen (NumPy array) capturada
        ret, frame = self.capture.read()
        
        # Si la cámara falla o se desconecta, ret será False
        if not ret:
            print("Error: No se pudo obtener el frame de la cámara.")
            return None
            
        # Retorna únicamente la imagen (frame)
        return frame

    def release(self):
        """
        Libera el recurso de la cámara para que otras aplicaciones puedan usarla.
        """
        self.capture.release()
