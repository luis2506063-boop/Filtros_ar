import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection
from filters.filtro_nariz import FiltroNariz
from filters.mustache_filter import MustacheFilter

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
        return

    # Inicializar detector y filtros
    print("Cargando modelo y filtros...")
    try:
        detector = FaceDetection()
        filtro_nariz = FiltroNariz(detector)
        
        # Instanciamos el filtro del bigote apuntando a tu carpeta y archivo
        filtro_bigote = MustacheFilter(
            facedetector=detector,
            png_path="assets/image.png", 
            offset_x=0,  # <--- AQUÍ ESTÁ LA LÍNEA AGREGADA
            offset_y=35, # Distancia hacia abajo desde la punta de la nariz
            target=130   # Ancho del bigote
        )
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

            face_landmarks_list = detector.detect_faces(frame)
            annotated_frame = frame.copy()
            
            if face_landmarks_list:
                for face_landmarks in face_landmarks_list:
                    # Aplica el filtro del bigote
                    annotated_frame = filtro_bigote.apply(annotated_frame, face_landmarks)
                    
                    # Si quieres la nariz de payaso al mismo tiempo, quítale el '#' a la línea de abajo
                    # annotated_frame = filtro_nariz.apply(annotated_frame, face_landmarks)
            
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