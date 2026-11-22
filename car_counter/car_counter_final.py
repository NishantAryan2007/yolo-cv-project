"""
car_counter_final.py
Stage 3: YOLOv8 detection + SORT tracking + line-crossing counting logic,
with per-vehicle-class breakdown (car / truck / bus / motorbike).
Visuals cleaned up: thinner boxes, explicit colors (no default pink).
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
from constants import COCO_CLASS_NAMES, VEHICLE_CLASSES
from helpers import calculate_iou

# CONFIGURATION

VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "highway.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.3
WINDOW_NAME = "Car Counter"

LINE_Y = 1000
LINE_X_START = 350
LINE_X_END = 710

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
counted_ids = set()
id_to_class = {}
class_counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 405, 720)

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended.")
        break

    results = model(frame, stream=True)

    detections = np.empty((0, 5))
    detection_classes = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            class_name = COCO_CLASS_NAMES[cls_id] if cls_id < len(COCO_CLASS_NAMES) else str(cls_id)

            if class_name in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))
                detection_classes.append(class_name)

    tracker_results = tracker.update(detections)

    cv2.line(frame, (LINE_X_START, LINE_Y), (LINE_X_END, LINE_Y), (0, 0, 255), 3)

    for result in tracker_results:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        if obj_id not in id_to_class:
            best_iou = 0
            best_class = "vehicle"
            for i, det_box in enumerate(detections):
                iou = calculate_iou([x1, y1, x2, y2], det_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_class = detection_classes[i]
            id_to_class[obj_id] = best_class

        obj_class = id_to_class.get(obj_id, "vehicle")

        # Thinner, sleeker corner box
        cvzone.cornerRect(frame, (x1, y1, w, h), l = 10, t = 2, rt = 0, colorC = (255, 255, 0))

        # Label — explicit dark background, white text (was defaulting to pink)
        cvzone.putTextRect(frame, f'{obj_class} ID {obj_id}', (max(0, x1), max(35, y1 - 10)),
                            scale = 1, thickness = 1, offset = 3,
                            colorR = (50, 50, 50), colorT = (255, 255, 255))

        # Center point dot — cyan, smaller
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), cv2.FILLED)

        if LINE_X_START < cx < LINE_X_END and LINE_Y - 15 < cy < LINE_Y + 15:
            if obj_id not in counted_ids:
                counted_ids.add(obj_id)
                if obj_class in class_counts:
                    class_counts[obj_class] += 1
                cv2.line(frame, (LINE_X_START, LINE_Y), (LINE_X_END, LINE_Y), (0, 255, 0), 5)

    # Total + per-class counter 
    y_offset = 50
    cvzone.putTextRect(frame, f'Total: {len(counted_ids)}', (50, y_offset), scale = 2, thickness = 2,
                        offset = 10, colorR = (50, 50, 50), colorT = (0, 255, 255))
    for i, (cls, count) in enumerate(class_counts.items()):
        cvzone.putTextRect(frame, f'{cls}: {count}', (50, y_offset + 50 + i * 40), scale = 1.5, thickness = 2,
                            offset = 8, colorR = (50, 50, 50), colorT = (255, 255, 255))

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)