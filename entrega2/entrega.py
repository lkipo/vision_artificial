import cv2 as cv
import numpy as np
# Lemos a imaxe
image = cv.imread('./barco.jpg')

# Función callback. Non fai nada en este caso
def nothing(x):
    pass

# Creamos ventana 
cv.namedWindow('deslizadores')
# cv.resizeWindow('deslizadores', 100, 300) # CAMBIAR ESTO

# Creamos trackbars
cv.createTrackbar('Hue_min', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Hue_max', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Saturation_min', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Saturation_max', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Value_min', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Value_max', 'deslizadores', 0, 255, nothing)

# Bucle principal
while(1):
    cv.imshow('image', image) # NECESITA MODIFICACIÓN
    
    # Xestión do bucle 
    k = cv.waitKey(1) & 0xFF
    if k == 115:
        break
    
    # obter posicións actuais dos trackbars
    
    hmin = cv.getTrackbarPos('Hue_min', 'deslizadores')
    hmax = cv.getTrackbarPos('Hue_max', 'deslizadores')
    satmin = cv.getTrackbarPos('Saturation_min', 'deslizadores')
    satmax = cv.getTrackbarPos('Saturation_max', 'deslizadores')
    valmin = cv.getTrackbarPos('Value_min', 'deslizadores')
    valmax = cv.getTrackbarPos('Value_max', 'deslizadores')

    # Creamos copia da imaxe en hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Creamos máscara
    mask = cv.inRange(image, hmin, hmax)

    res = cv.bitwise_and(image, image, mask=mask)