import cv2 as cv
import funcion as fn
import pandas as pd
import numpy as np


path_to_imgs = '../planeDataset/randGen/'

descritores = []
for i in range(128):
    print(i)
    image = cv.imread(path_to_imgs + str(i) + ".png", 0)
    ret, thresh = cv.threshold(image, 127, 255, 0)
    
    if image.size == None:
        print('erro na lectura', path_to_imgs + str(i) + ".png")
        exit(1)
    
    cv.imshow('ventana', image)
    # cv.waitKey(10)
    descritores.append(fn.fourierDesc(thresh, 10))
    
print(np.squeeze(descritores))
desc_df = pd.DataFrame(descritores)
desc_df.to_csv('algo.csv', index=False)
    