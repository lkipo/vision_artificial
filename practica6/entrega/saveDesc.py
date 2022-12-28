import cv2 as cv
import funcion as fn
import pandas as pd
import numpy as np


path_to_imgs = '../planeDataset/randGen/'

descritores = []
n_descritores = 9
for i in range(2048):
    # print(i)
    image = cv.imread(path_to_imgs + str(i) + ".png", 0)
    ret, thresh = cv.threshold(image, 127, 255, 0)
    
    if image.size == None:
        print('erro na lectura', path_to_imgs + str(i) + ".png")
        exit(1)
    
    # cv.imshow('ventana', image)
    # cv.waitKey(10)
    descritores.append(np.append(fn.fourierDesc(thresh, n_descritores), i//256))
    
    # chapuza incoming
    
if __name__=='__main__':  
    # print(np.squeeze(descritores))
    labels = ['X' + str(i) for i in range((n_descritores*4)-3)]
    labels.append('Y')
    desc_df = pd.DataFrame(descritores, columns=labels)
    desc_df.to_csv('algo.csv', index=False)    