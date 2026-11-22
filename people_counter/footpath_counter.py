"""
footpath_counter.py
YOLOv8 detection + SORT tracking + simple single-line counting
for a footpath video with one-directional (top-to-bottom) foot traffic.
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
from constants import COCO_CLASS_NAMES


# CONFIGURATION

VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "footpath.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8s.pt")

CONFIDENCE_THRESHOLD = 0.25  
WINDOW_NAME = "People Counter - Footpath"

LINE_Y = 1000
LINE_X_START = 0
LINE_X_END = 720


# LOAD MODEL, VIDEO, TRACKER

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
counted_ids = set()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 405, 720)


# MAIN LOOP

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended.")
        break

    results = model(frame, stream = True, verbose = False)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            class_name = COCO_CLASS_NAMES[cls_id] if cls_id < len(COCO_CLASS_NAMES) else str(cls_id)

            if class_name == "person" and conf > CONFIDENCE_THRESHOLD:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

    tracker_results = tracker.update(detections)

    cv2.line(frame, (LINE_X_START, LINE_Y), (LINE_X_END, LINE_Y), (0, 0, 255), 3)

    for result in tracker_results:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y2

        cvzone.cornerRect(frame, (x1, y1, w, h), l = 10, t = 2, rt = 0, colorC = (255, 255, 0))
        cvzone.putTextRect(frame, f'ID {obj_id}', (max(0, x1), max(35, y1 - 10)),
                            scale = 1, thickness = 1, offset = 3,
                            colorR = (50, 50, 50), colorT = (255, 255, 255))
        cv2.circle(frame, (cx, cy), 4, (255, 255, 0), cv2.FILLED)

        if LINE_X_START < cx < LINE_X_END and LINE_Y - 30 < cy < LINE_Y + 30:   
            if obj_id not in counted_ids:
                counted_ids.add(obj_id)
                cv2.line(frame, (LINE_X_START, LINE_Y), (LINE_X_END, LINE_Y), (0, 255, 0), 5)
                print(f"COUNTED -- ID {obj_id}")

    cvzone.putTextRect(frame, f'Count: {len(counted_ids)}', (50, 50), scale = 2, thickness = 2,
                        offset = 10, colorR = (50, 50, 50), colorT = (0, 255, 255))

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)