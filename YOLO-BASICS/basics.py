from ultralytics import YOLO
import cv2
model = YOLO('../yolo-weights/yolov8l.pt')
results= model("Images/2.jpg", show=True)

cv2.waitKey(0)