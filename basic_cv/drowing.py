import cv2 as cv
import numpy as np
blanck=np.zeros((500,500,3),dtype='uint8')
img=cv.imread('picture\Screenshot (70).png')
# cv.imshow('black',blanck)
blanck[200:205,200:300]=0,0,200
# cv.imshow('colour',blanck)
cv.circle(blanck,(250,250),100,(0,0,200),thickness=2)
# cv.imshow("circle",blanck)
gray=cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
cv.imshow("gray",gray)
blur=cv.GaussianBlur(img,(9,9),23)
cv.imshow("blur",blur)
img_cr=img[200:300,300:400]
cv.imshow('img_cr',img_cr)


cv.waitKey(0)