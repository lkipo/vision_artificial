import cv2 as cv
import funcion as fn
import pandas as pd
import numpy as np


path_to_imgs = '../planeDataset/randGen/'

descritores = []
for i in range(512):
    print(i)
    image = cv.imread(path_to_imgs + str(i) + ".png", 0)
    ret, thresh = cv.threshold(image, 127, 255, 0)
    
    if image.size == None:
        print('erro na lectura', path_to_imgs + str(i) + ".png")
        exit(1)
    
    cv.imshow('ventana', image)
    # cv.waitKey(10)
    descritores.append(np.append(fn.fourierDesc(thresh, 2), i//64))
    
    # chapuza incoming
    
    
print(np.squeeze(descritores))
desc_df = pd.DataFrame(descritores, columns=['X0', 'X1', 'X2', 'X3', 'X4', 'Y'])
# desc_df = pd.DataFrame(descritores, columns=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'Y'])
desc_df.to_csv('algo.csv', index=False)
    