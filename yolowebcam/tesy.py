from ultralytics import YOLO
import cv2
import cvzone
import math
import torch

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load the YOLO model and check if it is on GPU
model = YOLO("../yoloweights/yolov8l.pt")

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)  # Move model to the selected device (GPU or CPU)

print(f"Model is running on device: {device}")

classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            # Bounding box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class Names
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{conf} {classNames[cls]}', (x1, y1 - 20))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
