"""
car_counter.py
Stage 1: Base YOLOv8 detection on highway video : no tracking, no counting yet.
"""

import os
import cv2
import cvzone
import math
from ultralytics import YOLO


# CONFIGURAION OF PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "highway1.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.3
WINDOW_NAME = "Car Counter - Stage 1"

# Only these COCO classes matter for a car counter
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# LOAD MODEL + VIDEO

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Create a resizable window so it fits your screen
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


# MAIN LOOP

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended.")
        break

    results = model(frame, stream = True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

            if class_name in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                # Sleek corner-only box, cyan color, no connecting rectangle
                cvzone.cornerRect(frame, (x1, y1, w, h), l = 15, t = 3, rt = 0 , colorC = (255, 255, 0))
                cvzone.putTextRect(
                    frame, f'{class_name} {conf}',
                    (max(0, x1), max(35, y1 - 10)),
                    scale=1, thickness=1, offset=3
                )

    cv2.imshow(WINDOW_NAME, frame)

    # Exit if 'q' pressed OR window closed via the X button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # flushes pending GUI events so the window actually closes