import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/people.mp4")
model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsup = [103, 161, 296, 161]
limitsdown = [527, 489, 735, 489]
totalCountup = []
totalCountdown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsup[0] < cx < limitsup[2] and limitsup[1] - 15 < cy < limitsup[1] + 15:
            if totalCountup.count(id) == 0:
                totalCountup.append(id)
                cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0, 255, 0), 5)

        if limitsdown[0] < cx < limitsdown[2] and limitsdown[1] - 15 < cy < limitsdown[1] + 15:
            if totalCountdown.count(id) == 0:
                totalCountdown.append(id)
                cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCountup)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 199, 75), 8)
    cv2.putText(img, str(len(totalCountdown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 8)
    cv2.imshow("Image", img)
    cv2.waitKey(1)