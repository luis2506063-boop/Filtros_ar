import cv2
import numpy as np
from face_detector import FaceDetection

def test_face_detector():
    # Crear una imagen negra vacía
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detector = FaceDetection()
    
    print("Probando detección en imagen sin rostros...")
    # Esto debería retornar None y manejarlo correctamente sin errores
    result_image = detector.draw_face(dummy_image)
    
    print("Prueba completada: El código manejó la ausencia de rostros correctamente.")
    # Si llegamos aquí sin errores, la prueba pasó
    assert np.array_equal(result_image, dummy_image), "La imagen resultante debería ser igual a la original si no hay rostros"
    print("Validación exitosa.")

if __name__ == "__main__":
    test_face_detector()
