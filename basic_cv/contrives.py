import cv2 as cv
import numpy as np
img=cv.imread('picture\maxresdefault.jpg')
cv.imshow('abd',img)
blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
contrive = cv.Canny(blur,125,255)
cv.imshow('contrive1',contrive)
gray=cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
rat,thri=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('contrive2',thri)
contrivs,heric=cv.findContours(thri,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
blanck=np.zeros(img.shape,dtype="uint8")
blanck[:]=0,20,0
cv.drawContours(blanck,contrivs,-1,(0,0,255),1)
cv.imshow('draw',blanck)
print(len(contrivs))
cv.waitKey(0)