import cv2 as cv
import numpy as np
# Lemos a imaxe
image = cv.imread('../barco.jpg')

# Función callback. Non fai nada en este caso
def nothing(x):
    pass

# Creamos ventana 
cv.namedWindow('deslizadores')
# cv.resizeWindow('deslizadores', 100, 300) # CAMBIAR ESTO

# Creamos trackbars
cv.createTrackbar('Hue_min', 'deslizadores', 0, 180, nothing) # Cambiar o valor máximo a 255 para traballar en espazos distintos a HSV
cv.createTrackbar('Hue_max', 'deslizadores', 0, 180, nothing) # Cambiar o valor máximo a 255 para traballar en espazos distintos a HSV
cv.createTrackbar('Saturation_min', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Saturation_max', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Value_min', 'deslizadores', 0, 255, nothing)
cv.createTrackbar('Value_max', 'deslizadores', 0, 255, nothing)

# Bucle principal
while(1):
    
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
    # hsv = cv.cvtColor(image, cv.COLOR_BGR2LAB) # Descomentar para filtrar en Lab
    # hsv = image # Descomentar para filtrar en BGR

    # Creamos máscara
    mask = cv.inRange(hsv, np.array([hmin, satmin, valmin]), np.array([hmax, satmax, valmax]))

    res = cv.bitwise_and(image, image, mask=mask)
    
    cv.imshow('bitwise', res)
    cv.imshow('mascara', mask)
