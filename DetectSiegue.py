import cv2
import numpy as np


def DetectSiegue(inImage) -> np.ndarray:
    hsv = cv2.cvtColor(inImage, cv2.COLOR_BGR2HSV)

    # Definir un rango de tonos de verde con ligeras variaciones
    lower_green = np.array([30, 30, 30])  # Umbral inferior para el verde
    upper_green = np.array([90, 255, 255])  # Umbral superior para el verde

    # Crear máscaras utilizando la umbralización para detectar variaciones en el tono de verde
    mask1 = cv2.inRange(hsv, lower_green, upper_green)

    # Invertir la máscara para encontrar áreas que no coincidan exactamente con el rango definido
    mask2 = cv2.bitwise_not(mask1)

    # Combinar las máscaras para obtener las áreas donde el verde tiene ligeras variaciones
    mask_combined = cv2.bitwise_and(mask1, mask2)

    # Encontrar los contornos en la imagen segmentada
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en una copia de la imagen original
    areas_image = inImage.copy()
    cv2.drawContours(areas_image, contours, -1, (0, 255, 0), 2)

    # Mostrar la imagen con los contornos de las áreas con variaciones leves en el tono de verde
    cv2.imshow('Áreas con variaciones leves en el tono de verde', areas_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return areas_image

def DetectSiegue1(inImage) -> np.ndarray:
    def nothing(x):
        pass
    cv2.namedWindow('Imagen')
    # Crear trackbars para ajustar los umbrales
    cv2.createTrackbar('Hue Min', 'Imagen', 0, 255, nothing)
    cv2.createTrackbar('Hue Max', 'Imagen', 0, 255, nothing)
    cv2.createTrackbar('Saturation Min', 'Imagen', 0, 255, nothing)
    cv2.createTrackbar('Saturation Max', 'Imagen', 0, 255, nothing)
    cv2.createTrackbar('Value Min', 'Imagen', 0, 255, nothing)
    cv2.createTrackbar('Value Max', 'Imagen', 0, 255, nothing)

    while True:
        # Convertir la imagen de BGR a HSV
        hsv = cv2.cvtColor(inImage, cv2.COLOR_BGR2HSV)

        # Obtener los valores actuales de los trackbars
        hue_min = cv2.getTrackbarPos('Hue Min', 'Imagen')
        hue_max = cv2.getTrackbarPos('Hue Max', 'Imagen')
        saturation_min = cv2.getTrackbarPos('Saturation Min', 'Imagen')
        saturation_max = cv2.getTrackbarPos('Saturation Max', 'Imagen')
        value_min = cv2.getTrackbarPos('Value Min', 'Imagen')
        value_max = cv2.getTrackbarPos('Value Max', 'Imagen')

        # Definir los rangos de color en HSV con los valores actuales
        lower_green = np.array([hue_min, saturation_min, value_min])
        upper_green = np.array([hue_max, saturation_max, value_max])

        # Crear una máscara utilizando la umbralización
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar la máscara a la imagen original
        result = cv2.bitwise_and(inImage, inImage, mask=mask)

        # Mostrar la imagen resultante
        cv2.imshow('Resultado', result)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.destroyAllWindows()
    return 0