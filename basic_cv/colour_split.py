import cv2 as cv
import matplotlib.pyplot as plot
img=cv.imread("../picture/maxresdefault.jpg")
cv.imshow('orgi',img)
plot.imshow(img)
plot.show()
bgr_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('RGB',bgr_rgb)
b,g,r=cv.split(img)
cv.imshow('blue',b)
cv.imshow('red',r)
cv.waitKey(0)
