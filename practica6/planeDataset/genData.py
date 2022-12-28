import cv2 as cv
from random import randint, uniform

def getMask(img):
    img = cv.bitwise_not(img)
    ret, thresh = cv.threshold(img, 127, 255, 0)

    return thresh

def genImage(img, angle, center, scale, size = (800, 800)):
    img = cv.resize(img, size)
    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    rot = cv.warpAffine(img, rot_mat, size)
    # cv.imshow('ventana', rot)
    # cv.waitKey(100)
    
    return rot
    
if __name__=='__main__':
    count = 0
    for i in range(8):
        image = cv.imread('plane' + str(i) + '.png')
        mask = getMask(image)
        for j in range(1024):
            angle = randint(0, 359)
            scale = uniform(0.2, 1)
            # print(scale)
            # x = randint(((-1+scale)*400)//1, ((1-scale*400))//1) + 400
            # y = randint(((-1+scale)*400)//1, ((1-scale*400))//1) + 400
            # x = randint(200, 600)
            # y = randint(200, 600)
            x = randint((400*scale)//1, 800-(400*scale)//1)
            y = randint((400*scale)//1, 800-(400*scale)//1)
            transf = genImage(mask, angle, (x, y), scale)
            cv.imwrite("randGen/" + str(count)+ ".png", transf)
            count += 1