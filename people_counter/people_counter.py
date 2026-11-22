"""
people_counter.py
Final attempt — escalator video, front-angle, using yolov8s.
Y=600, three equal-width zones, box-center dot, normal playback speed.
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

VIDEO_PATH = os.path.join(BASE_DIR, "..", "media", "videos", "escalator.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8s.pt")

CONFIDENCE_THRESHOLD = 0.3
WINDOW_NAME = "People Counter - Escalators"

ESCALATOR_ZONES = [
    {"name": "Escalator 1", "direction": "up",   "line_y": 600, "x_start": 10,  "x_end": 230},
    {"name": "Escalator 2", "direction": "up",   "line_y": 600, "x_start": 250, "x_end": 450},
    {"name": "Escalator 3", "direction": "down", "line_y": 600, "x_start": 470, "x_end": 660},
]


# LOAD MODEL, VIDEO, TRACKER

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
counted_ids_per_zone = {zone["name"]: set() for zone in ESCALATOR_ZONES}
direction_counts = {"up": 0, "down": 0}

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

    for zone in ESCALATOR_ZONES:
        color = (0, 255, 255) if zone["direction"] == "up" else (0, 165, 255)
        cv2.line(frame, (zone["x_start"], zone["line_y"]), (zone["x_end"], zone["line_y"]), color, 3)

    for result in tracker_results:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2   

        cvzone.cornerRect(frame, (x1, y1, w, h), l = 10, t = 2, rt = 0, colorC = (255, 255, 0))
        cvzone.putTextRect(frame, f'ID {obj_id}', (max(0, x1), max(35, y1 - 10)),
                            scale = 1, thickness = 1, offset = 3,
                            colorR = (50, 50, 50), colorT = (255, 255, 255))
        cv2.circle(frame, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

        for zone in ESCALATOR_ZONES:
            if zone["x_start"] < cx < zone["x_end"] and zone["line_y"] - 20 < cy < zone["line_y"] + 20:
                if obj_id not in counted_ids_per_zone[zone["name"]]:
                    counted_ids_per_zone[zone["name"]].add(obj_id)
                    direction_counts[zone["direction"]] += 1
                    print(f"COUNTED -- {zone['name']} ({zone['direction']}) | ID {obj_id}")

    cvzone.putTextRect(frame, f'Up: {direction_counts["up"]}', (50, 50), scale = 2, thickness = 2,
                        offset = 10, colorR = (50, 50, 50), colorT = (0, 255, 255))
    cvzone.putTextRect(frame, f'Down: {direction_counts["down"]}', (50, 110), scale = 2, thickness = 2,
                        offset = 10, colorR = (50, 50, 50), colorT = (0, 165, 255))

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)