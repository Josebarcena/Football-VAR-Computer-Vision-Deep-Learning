import cv2
from PIL import Image
import numpy as np

def DetectField1(inImage):
    hsv = cv2.cvtColor(inImage, cv2.COLOR_BGR2HSV)

    # Definir rango de color verde en HSV
    lower_green = np.array([25, 40, 20])  # Ajusta estos valores para el rango de verde deseado
    upper_green = np.array([80, 255, 255])  # Ajusta estos valores para el rango de verde deseado

    # Crear una máscara para el rango de color verde
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Aplicar la máscara a la imagen original para obtener solo el campo de fútbol verde
    result = cv2.bitwise_and(inImage, inImage, mask=mask)

    # Obtener contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)

    # Crear una máscara solo con el contorno más grande
    mask_contour = np.zeros_like(mask)
    cv2.drawContours(mask_contour, [largest_contour], -1, 255, -1)
    
    kernel = np.ones((2, 2), np.uint8)
    # Dilatar el contorno más grande para capturar más elementos
    dilated_mask_contour = cv2.dilate(mask_contour, kernel, iterations=2)
    
    # Aplicar la máscara dilatada a la imagen original para conservar los elementos que se extienden fuera del contorno
    result_dilated = cv2.bitwise_and(inImage, inImage, mask=dilated_mask_contour)

    # Combinar la máscara dilatada con la máscara original
    result_combined = cv2.bitwise_or(result, result_dilated)

    return result_combined

def DetectField(inImage):
        
    hsv = cv2.cvtColor(inImage, cv2.COLOR_BGR2HSV)

    # Definir rango de color verde en HSV
    lower_green = np.array([25, 40, 20])  # Ajusta estos valores para el rango de verde deseado
    upper_green = np.array([80, 255, 255])  # Ajusta estos valores para el rango de verde deseado

    # Definir rango de color blanco en HSV
    lower_white = np.array([0, 0, 200])  # Ajusta estos valores para el rango de blanco deseado
    upper_white = np.array([180, 40, 255])  # Ajusta estos valores para el rango de blanco deseado
    # Crear una máscara para el rango de color verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_green, mask_white)

    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Aplicar la máscara a la imagen original para obtener solo el campo de fútbol verde
    result = cv2.bitwise_and(inImage, inImage, mask=mask)
    
    
    imagen_objeto_gris = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, objeto_mascara = cv2.threshold(imagen_objeto_gris, 200, 255, cv2.THRESH_BINARY)

    # Invertir la máscara para tener el objeto en blanco y el fondo en negro
    objeto_mascara = cv2.bitwise_not(objeto_mascara)
    # Aplicar la máscara del objeto a la imagen con fondo para extraer el objeto
    result = cv2.bitwise_and(inImage, inImage, mask=objeto_mascara)


    mask_blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    
    contours, _ = cv2.findContours(mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encuentra el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    mask_contour = np.zeros_like(mask)
    cv2.drawContours(mask_contour, [largest_contour], -1, 255, -1)

    # Aplicar la máscara del contorno a la imagen original para conservar solo el objeto marcado
    result = cv2.bitwise_and(inImage, inImage, mask=mask_contour)

    return result


