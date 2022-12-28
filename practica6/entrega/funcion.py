import cv2 as cv
import numpy as np
from pyefd import elliptic_fourier_descriptors

### NOTAS IMPORTANTES:
#       funciona en caso de que so haxa un contorno, en caso contrario modificar liñas 15+23
#       propenso a fallos se a imaxe é de negro sobre branco e non de branco sobre negro
def radialFunc(img):
    # COMENTAR ESTO
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, 0, 120, 2)
    cv.imshow('ventana', img)
    # cv.waitKey(3000)
    # print(contours)d


    M = cv.moments(contours[0]) # non me fío de poder usar esto
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    centroid = (cx, cy)
    distances = []
    
    for point in contours[0]:
        x = np.power((point[0][0] - centroid[0]), 2)
        y = np.power((point[0][1] - centroid[1]), 2)
        distances.append(np.power(x+y, 0.5))
        
    return distances  

def contourDesc(img):
    pass
    
def fourierDesc(img, odr):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    coeffs = elliptic_fourier_descriptors(np.squeeze(contours[0]), order = odr, normalize=True)
    
    return coeffs.flatten()[3:]
    
if __name__=='__main__':
    image = cv.imread('test.png', 0)
    ret, thresh = cv.threshold(image, 127, 255, 0)
    inv = cv.bitwise_not(thresh)
    cv.imshow('ventana', inv)
    cv.waitKey(1000)
    print(fourierDesc(inv, 10))

