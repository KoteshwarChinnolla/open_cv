import cv2 as cv
import numpy as np
img=cv.imread('picture\Screenshot (70).png')
def transulate(img,x,y):
    mat=np.float32([[1,0,x],[0,1,y]])
    di=(img.shape[1],img.shape[0])
    return cv.warpAffine(img, mat, di)
trans=transulate(img,-100,-100)
cv.imshow('trans',trans)
#rotate
def rotate(img,ang,rotpoint=None):
    (hight,width)=img.shape[:2]
    if rotpoint is None:
        rotpoint=(hight//2,width//2)
    mat=cv.getRotationMatrix2D(rotpoint,ang,1.0)
    di=(hight,width)
    return cv.warpAffine(img,mat,di)
rot=rotate(img,45)
cv.imshow('rotate',rot)
cv.waitKey(0)