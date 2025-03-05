import cv2 as cv
import numpy as np
img=cv.imread('../picture/mab.jpg')
blur=cv.GaussianBlur(img,(9,9),0)
line=cv.Canny(blur,28,37,13)
gray=cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
thresh=cv.adaptiveThreshold(gray,200,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,9,3)
cv.imshow('mab',thresh)
cv.waitKey(0)