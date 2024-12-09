
from ultralytics import YOLO
import cv2
model=YOLO('../yoloweights/yolov8n.pt')
results = model("images/_SOM4670.JPG",show=True)

cv2.waitKeyEx(0)
 # Should return True

