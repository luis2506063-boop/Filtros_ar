import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection

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

    # Inicializar el detector de caras (MediaPipe 1.0.0)
    print("Cargando modelo de detección facial (MediaPipe Face Landmarker)...")
    try:
        detector = FaceDetection()
        print("Modelo cargado con éxito.")
    except Exception as e:
        print(f"\n[ERROR] Error al inicializar el detector de rostros: {e}")
        camera.release()
        return

    # Crear ventana de visualización interactiva
    window_name = "Filtros AR - Malla Facial interactiva"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nIniciando transmisión en vivo. Ajusta el tamaño de la ventana si lo deseas.")
    try:
        while True:
            # Capturar el frame de la cámara
            frame = camera.get_frame()
            if frame is None:
                print("[ERROR] No se pudo obtener el frame de la cámara. Saliendo...")
                break

            # Dibujar la malla facial sobre el frame actual en tiempo real
            annotated_frame = detector.draw_face(frame)

            # Mostrar el frame procesado en la ventana
            cv2.imshow(window_name, annotated_frame)

            # Escuchar las teclas: esperar 1 ms.
            # 0xFF limpia el valor del entero para quedarse con el código ASCII de la tecla
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 27 es la tecla ESC en ASCII
                print("\nCerrando transmisión de video por solicitud del usuario...")
                break
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado durante la ejecución: {e}")
    finally:
        # Liberación limpia de recursos al salir
        camera.release()
        cv2.destroyAllWindows()
        print("Cámara liberada y ventanas cerradas. ¡Aplicación finalizada correctamente!")

if __name__ == "__main__":
    main()
