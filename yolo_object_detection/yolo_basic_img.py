from ultralytics import YOLO
import cv2 as cv

model=YOLO("../Yolo-Weights/yolov8n.pt")
results = model("picture/people.jpg", show=True)

cv.waitKey(0)