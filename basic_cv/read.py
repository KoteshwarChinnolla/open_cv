import cv2 as cv
#img=cv.imread('picture\Screenshot (70).png')
#cv.imshow('Screenshot (70).png',img)
video=cv.VideoCapture('..\picture2\yt1s.com - Just How Fast Night Changes  shorts.mp4')
while(True):
    isTrue,read=video.read()
    cv.imshow("Video",read)
    resized=cv.resize(read,(int(read.shape[1]*0.5),int(read.shape[0]*0.5)),interpolation=cv.INTER_AREA)
    cv.imshow("resized",resized)
    if(cv.waitKey(20) & 0xFF==ord('d')):
        break

video.release()
cv.destroyAllWindows()