import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetection:
    def __init__(self, 
                 max_num_faces=1, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        # Ruta al archivo del modelo .task que descargamos previamente en la terminal
        model_path = 'face_landmarker.task' 
        
        # Configuramos las opciones base apuntando a nuestro modelo
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Configuramos las opciones del landmarker
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Mantenemos IMAGE para compatibilidad con tu código actual
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Inicializamos el modelo oficial de MediaPipe
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Rescatamos las utilidades de dibujo de MediaPipe 1.0.0
        self.drawing_utils = vision.drawing_utils
        self.face_landmarks_connections = vision.FaceLandmarksConnections

    def detect_faces(self, image):
        """
        Detecta los rostros y devuelve los puntos clave (landmarks).
        """
        # Convertimos la imagen de BGR (formato nativo de OpenCV) a RGB
        # y la pasamos al formato estricto que exige MediaPipe (mp.Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Procesamos la imagen para encontrar los rostros
        results = self.landmarker.detect(mp_image)
        
        # Retornamos la lista de rostros detectados
        return results.face_landmarks

    def landmark_to_coordinates(self, image, landmark):
        """
        Convierte un landmark normalizado (0.0 - 1.0) a coordenadas 
        de píxeles reales (x, y) basadas en el tamaño de la imagen.
        """
        alto, ancho, _ = image.shape
        x = int(landmark.x * ancho)
        y = int(landmark.y * alto)
        return x, y

    def draw_face(self, image):
        """
        Dibuja la malla facial sobre la imagen original.
        """
        # 1. Ejecutamos la detección en el fotograma actual
        face_landmarks_list = self.detect_faces(image)
        
        # 2. Validación de seguridad: Si no hay rostros detectados, devolvemos el video tal cual
        if not face_landmarks_list:
            return image
            
        # 3. Creamos una copia de la imagen para no sobreescribir la original de forma destructiva
        annotated_image = image.copy()
        
        # 4. Recorremos cada rostro detectado en la pantalla
        for face_landmarks in face_landmarks_list:
            
            # En la nueva API de MediaPipe 1.0.0, la función draw_landmarks acepta 
            # directamente la lista de NormalizedLandmarks sin traducción a Protobuf.
            self.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.face_landmarks_connections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None, 
                connection_drawing_spec=self.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
        # Retornamos la imagen ya con la malla dibujada en color verde
        return annotated_image