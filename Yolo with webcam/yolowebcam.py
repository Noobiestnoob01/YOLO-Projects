from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#to start capturing the image from webcam
cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
#cap = cv2.VideoCapture("../Videos/cars.mp4")  # For Video
#to use the yolo model
model = YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","comb","bottle"
              ]
prev_frame_time = 0
new_frame_time = 0
#while the code is closed
while True:
    new_frame_time = time.time()
    success, img=cap.read()
    results=model(img,stream=True)
    #To print boxes
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #for the dimensions of the box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # print(x1,y1,x2,y2)

            #-> CHOOSE ANY ONE BETWEEN ABOVE AND BELOW OPTION

            w, h = x2-x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            #confidence Value
            conf= math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(40,y1)))

            #Class Name
            cls=int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)))#,scale=1,thickness=2 if we want to adjust that
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
