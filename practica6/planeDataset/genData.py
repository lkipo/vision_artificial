import cv2 as cv
from random import randint, random

def getMask(img):
    img = cv.bitwise_not(img)
    ret, thresh = cv.threshold(img, 127, 255, 0)
    inv = cv.bitwise_not(thresh)
    cv.imshow('ventadddna', thresh)
    cv.waitKey(100)
    return thresh

def genImage(img, angle, center, scale, size = (800, 800)):
    img = cv.resize(img, size)
    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    rot = cv.warpAffine(img, rot_mat, size)
    cv.imshow('ventana', rot)
    cv.waitKey(100)
    
    return rot
    
if __name__=='__main__':
    for i in range(8):
        image = cv.imread('plane' + str(i) + '.png')
        mask = getMask(image)
        for j in range(16):
            angle = randint(0, 359)
            scale = random()
            print(scale)
            # x = randint(((-1+scale)*400)//1, ((1-scale*400))//1) + 400
            # y = randint(((-1+scale)*400)//1, ((1-scale*400))//1) + 400
            # x = randint(200, 600)
            # y = randint(200, 600)
            x = randint((400*scale)//1, 800-(400*scale)//1)
            y = randint((400*scale)//1, 800-(400*scale)//1)
            transf = genImage(mask, angle, (x, y), scale)
            cv.imwrite(str(i)+str(j)+".jpg", transf)