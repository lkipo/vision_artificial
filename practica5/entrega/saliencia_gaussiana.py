import cv2 as cv
import numpy as np
from metricas_eval_sal import *

def gaussianSalency(img):
    gauss = np.random.nor

if __name__ == '__main__':

    path_to_images = "Eval_saliencia/dataset_Toronto/Imaxes/"
    datalen = 120
    for index in range(datalen):
        
        image = cv.imread(path_to_images + str(index+1) + ".jpg")
        
        gaussianSalency(image)