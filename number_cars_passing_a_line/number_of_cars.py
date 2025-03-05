import cv2 as cv
from ultralytics import YOLO
import cvzone

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

model=YOLO("../Yolo-Weights/yolov8n.pt")
video=cv.VideoCapture('../picture/cars.mp4')
mask = cv.imread("../picture/mask.png")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while(True):
    isTrue,read=video.read()
    imgRegion = cv.bitwise_and(read, mask)


    results = model(imgRegion, show=True)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv.rectangle(read,(x1,y1),(x2,y2),(0,255,0),2)

            w,h=x2-x1,y2-y1
            if box.conf[0]*100>30 and classNames[int(box.cls[0])]=='car':
                cvzone.cornerRect(read,(x1,y1,w,h))
                cvzone.putTextRect(read,f'{classNames[int(box.cls[0])]} {int(box.conf[0]*100)}%',(max(0,x1),max(35,y1)),scale=1,thickness=2,offset=5)

    cv.imshow("Video",read)
    cv.waitKey(1)


video.release()
cv.destroyAllWindows()

#Information returned by each box

# ultralytics.engine.results.Boxes object with attributes:

# cls: tensor([0.])
# conf: tensor([0.2853])
# data: tensor([[5.9101e+02, 2.6185e+02, 6.1161e+02, 3.1412e+02, 2.8529e-01, 0.0000e+00]])
# id: None
# is_track: False
# orig_shape: (720, 1280)
# shape: torch.Size([1, 6])
# xywh: tensor([[601.3117, 287.9875,  20.5969,  52.2653]])
# xywhn: tensor([[0.4698, 0.4000, 0.0161, 0.0726]])
# xyxy: tensor([[591.0132, 261.8549, 611.6102, 314.1201]])
# xyxyn: tensor([[0.4617, 0.3637, 0.4778, 0.4363]])