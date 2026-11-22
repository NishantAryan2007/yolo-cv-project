"""
car_counter_tracking.py
Stage 2: YOLOv8 detection + SORT tracking for persistent object IDs.
"""

import os
import sys
import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "utils"))
from sort import Sort

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "highway1.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.3
WINDOW_NAME = "Car Counter - Stage 2 (Tracking)"

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


# LOAD MODEL, VIDEO, TRACKER

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)


# MAIN LOOP

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended.")
        break

    results = model(frame, stream=True)

    # Build this frame's detections array for SORT
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

            if class_name in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

    # Feed detections to SORT, get back persistent IDs
    tracker_results = tracker.update(detections)

    # Draw using SORT's tracked output
    for result in tracker_results:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l = 15, t = 3, rt = 0, colorC = (255, 255, 0))
        cvzone.putTextRect(frame, f'ID {obj_id}', (max(0, x1), max(35, y1 - 10)), scale = 1, thickness = 1, offset = 3)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)