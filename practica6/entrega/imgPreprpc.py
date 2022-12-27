import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

path = './carDataset/crop/' ### CAMBIAR ---> poñer argumentos 

for i in range(9):
    
    imname = str(i) + ".jpeg"
    
    print(path+imname)
    image = cv.imread(path + imname, 0)
    
    # if image == None:
    #     print('Erro na lectura da imaxe. Comproba os paths')
    
    # proc = cv.
    
    resize = cv.resize(image, (800, 400))
    # tresh = cv.adaptiveThreshold(resize, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY)
    # th, dst = cv.threshold(src, thresh, maxValue, cv.THRESH_BINARY)
    img_thresh_adp = cv.adaptiveThreshold(resize, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 7)
    
    contours, hierarchy = cv.findContours(img_thresh_adp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    
    # cv.imshow('vtest', count)
    
    cont = cv.drawContours(resize, contours, -1, (0, 255, 0), 3)
    cv.imshow('ventana', cont)
    key = cv.waitKey(1000)