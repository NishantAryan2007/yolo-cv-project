"""
traffic_violation_detector.py
Detects motorcycle riders without helmets, tracks them with SORT, and
counts violations as they cross a hand-painted mask zone on the road.
Logs violations to CSV via the shared ViolationLogger.
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
from violation_logger import ViolationLogger

# CONFIGURATION

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "traffic_best_v2.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "traffic.mp4")
MASK_PATH = os.path.join(BASE_DIR, "..", "traffic_mask.png")

CONFIDENCE_THRESHOLD = 0.4
WINDOW_NAME = "Traffic Violation Detector"

TRAFFIC_CLASSES = ["With Helmet", "Without Helmet"]
CLASS_COLORS = {
    "With Helmet":    (0, 255, 0),
    "Without Helmet": (0, 0, 255),
}


# LOAD MODEL, VIDEO, MASK, TRACKER, LOGGER

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
mask = cv2.imread(MASK_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
if mask is None:
    raise RuntimeError(f"Could not load mask: {MASK_PATH}")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
logger = ViolationLogger(os.path.join(BASE_DIR, "traffic_violations_log.csv"), cooldown_seconds = 3)

counted_ids = set()
violation_count = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

# MAIN LOOP

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended.")
        break

    results = model(frame, stream = True, verbose = False)

    detections = np.empty((0, 5))
    detection_classes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_id = int(box.cls[0])
            class_name = TRAFFIC_CLASSES[cls_id] if cls_id < len(TRAFFIC_CLASSES) else str(cls_id)

            if class_name == "Without Helmet" and conf > CONFIDENCE_THRESHOLD:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
                detection_classes.append(class_name)

            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l = 8, t = 2, rt = 0, colorC = color)
            cvzone.putTextRect(frame, f'{class_name} {conf}', (max(0, x1), max(20, y1 - 8)),
                                scale = 0.6, thickness = 1, offset = 4,
                                colorR = (50, 50, 50), colorT = color)

    tracker_results = tracker.update(detections)

    for result in tracker_results:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y2

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            if mask[cy, cx][0] > 127:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    violation_count += 1
                    logger.log("Without Helmet - crossed zone", "HIGH")

    mask_overlay = cv2.addWeighted(frame, 0.8, mask, 0.2, 0)
    frame = mask_overlay

    cvzone.putTextRect(frame, f'Violations: {violation_count}', (20, 40), scale = 1.5, thickness = 2,
                        offset = 8, colorR = (50, 50, 50), colorT = (0, 0, 255))

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print(f"\n[INFO] Session ended. Total violations: {violation_count}")
print(f"[INFO] Log saved to: {logger.csv_path}")
