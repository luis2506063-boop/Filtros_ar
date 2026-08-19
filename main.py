import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection
from filters.filtro_nariz import FiltroNariz

def main():
    print("=========================================")
    print("   Iniciando aplicación de Filtros AR   ")
    print("=========================================")
    print("Instrucciones:")
    print("  - Presiona 'q' o la tecla 'ESC' para salir.")
    print("=========================================")

    # Inicializar el administrador de cámara
    try:
        camera = CameraManager(camera_index=0)
    except RuntimeError as e:
        print(f"\n[ERROR] No se pudo iniciar la cámara: {e}")
        print("Por favor, asegúrate de tener una cámara web conectada y funcional.")
        return

    # Inicializar el detector de caras y el filtro de nariz
    print("Cargando modelo y filtros...")
    try:
        detector = FaceDetection()
        filtro_nariz = FiltroNariz(detector)
        print("Carga completada con éxito.")
    except Exception as e:
        print(f"\n[ERROR] Error al inicializar: {e}")
        camera.release()
        return

    window_name = "Filtros AR - Rostro e Interacción"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            # 1. Detectar los rostros primero
            face_landmarks_list = detector.detect_faces(frame)

            # 2. Dibujar la malla facial (opcional, para ver los puntos)
            # Nota: detector.draw_face ya hace su propia detección, 
            # para ser eficientes, mejor dibujamos sobre el frame actual si hay landmarks
            annotated_frame = frame.copy()
            
            if face_landmarks_list:
                for face_landmarks in face_landmarks_list:
                    # Aplicar el filtro de la nariz (usa face_landmarks)
                    annotated_frame = filtro_nariz.apply(annotated_frame, face_landmarks)
                    
                    # Opcional: Dibujar la malla también para referencia
                    # (Podríamos mover el dibujo a una función separada si quisiéramos)
                    # detector.draw_face(frame) ya existe, pero aquí lo haremos manual 
                    # o usaremos el método para mostrar todo.
            
            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break
    except Exception as e:
        print(f"\n[ERROR]: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Aplicación finalizada.")

if __name__ == "__main__":
    main()
