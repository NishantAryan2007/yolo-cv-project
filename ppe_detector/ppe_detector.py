"""
ppe_detector.py
Hospital PPE Detector — per-person compliance checking with violation logging.
Runs two models: a pretrained COCO model (for person detection) and our
custom-trained ppe_best.pt (for PPE item detection), matches PPE items to
individual people using IoU overlap, and logs non-compliant sightings to CSV.

Controls:  q = quit    s = save a full-resolution screenshot
"""

import os
import sys
import cv2
import cvzone
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "utils"))
from constants import COCO_CLASS_NAMES
from violation_logger import ViolationLogger
from helpers import calculate_iou

# CONFIGURATION

PERSON_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "yolov8n.pt")
PPE_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ppe_best.pt")
# VIDEO_SOURCE = 0     # for webcam
VIDEO_SOURCE = os.path.join(BASE_DIR, "..", "media", "videos", "ppe_test1.mp4")  # for video
SCREENSHOT_PATH = os.path.join(BASE_DIR, "..", "docs", "screenshots", "ppe_detector.png")

CONFIDENCE_THRESHOLD = 0.4
PERSON_CONFIDENCE = 0.4
WINDOW_NAME = "Hospital PPE Detector - Per Person"
DISPLAY_HEIGHT = 700   # max on-screen window height in pixels

PPE_CLASSES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]
REQUIRED_ITEMS = ["Mask", "Gloves"]

CLASS_COLORS = {
    "Coverall":    (255, 255, 0),
    "Face_Shield": (0, 255, 0),
    "Gloves":      (0, 165, 255),
    "Goggles":     (255, 0, 255),
    "Mask":        (0, 255, 255),
}


# LOAD MODELS, VIDEO, LOGGER

person_model = YOLO(PERSON_MODEL_PATH)
ppe_model = YOLO(PPE_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
print(f"Video resolution: {width}x{height}")

logger = ViolationLogger(os.path.join(BASE_DIR, "ppe_violations_log.csv"), cooldown_seconds=5)


# DISPLAY SCALING
# The window shows the frame shrunk by display_ratio, so anything drawn on the
# frame is shrunk by the same amount. `ui` scales all drawing UP by the inverse,
# so text/boxes look the same size on screen regardless of video resolution.

display_ratio = min(1.0, DISPLAY_HEIGHT / height)
ui = 1.0 / display_ratio

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, int(width * display_ratio), int(height * display_ratio))


def draw_label(frame, text, x, y, text_color, base_scale = 0.8, base_thickness = 1, base_offset = 5):
    """Readable label with a dark background, scaled for the display, kept inside the frame."""
    scale = base_scale * ui
    thickness = max(1, int(round(base_thickness * ui)))
    offset = int(base_offset * ui)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, scale, thickness)

    x = int(max(0, min(x, frame.shape[1] - tw - 2 * offset - 1)))
    y = int(max(th + 2 * offset, y))

    cvzone.putTextRect(frame, text, (x, y), scale=scale, thickness=thickness,
                       offset=offset, colorR=(50, 50, 50), colorT=text_color)


print("Controls: q = quit, s = save full-resolution screenshot")


# MAIN LOOP

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] Video ended or webcam disconnected.")
        break

    person_results = person_model(frame, stream = True, verbose = False)
    ppe_results = ppe_model(frame, stream = True, verbose = False)

    person_boxes = []
    for r in person_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if COCO_CLASS_NAMES[cls_id] == "person" and conf > PERSON_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

    ppe_boxes = []
    detected_this_frame = set()
    for r in ppe_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = PPE_CLASSES[cls_id]
                ppe_boxes.append((class_name, x1, y1, x2, y2, conf))
                detected_this_frame.add(class_name)

    corner_len = int(10 * ui)
    line_thick = max(1, int(round(2 * ui)))

    # PPE item boxes, color-coded, with readable labels
    for class_name, ex1, ey1, ex2, ey2, conf in ppe_boxes:
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cvzone.cornerRect(frame, (ex1, ey1, ex2 - ex1, ey2 - ey1),
                          l = corner_len, t = line_thick, rt = 0, colorC = color)
        draw_label(frame, class_name, ex1, ey1 - int(8 * ui), color, base_scale = 0.75)

    # Per-person compliance box + label + violation logging
    for (px1, py1, px2, py2) in person_boxes:
        person_ppe = set()
        for class_name, ex1, ey1, ex2, ey2, conf in ppe_boxes:
            iou = calculate_iou([px1, py1, px2, py2], [ex1, ey1, ex2, ey2])
            if iou > 0.02:
                person_ppe.add(class_name)

        missing = [item for item in REQUIRED_ITEMS if item not in person_ppe]
        box_color = (0, 0, 255) if missing else (0, 255, 0)

        cvzone.cornerRect(frame, (px1, py1, px2 - px1, py2 - py1),
                          l=int(14 * ui), t = line_thick, rt = 0, colorC = box_color)

        status_text = f'Person - Missing: {", ".join(missing)}' if missing else 'Person - PPE OK'
        draw_label(frame, status_text, px1, py1 - int(10 * ui), box_color, base_scale = 0.9)

        if missing:
            violation_key = ", ".join(sorted(missing))
            severity = "HIGH" if "Mask" in missing else "MEDIUM"
            logger.log(violation_key, severity)

    # Left-side summary panel
    panel_x = int(15 * ui)
    y_offset = int(40 * ui)
    draw_label(frame, 'PPE Status', panel_x, y_offset, (255, 255, 255),
               base_scale = 1.0, base_thickness = 2)
    y_offset += int(36 * ui)
    for item in PPE_CLASSES:
        status = "Detected" if item in detected_this_frame else "Not visible"
        color = CLASS_COLORS[item] if item in detected_this_frame else (150, 150, 150)
        draw_label(frame, f'{item}: {status}', panel_x, y_offset, color, base_scale = 0.8)
        y_offset += int(32 * ui)

    cv2.imshow(WINDOW_NAME, frame)

    # One waitKey call per frame — calling it twice can swallow key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        os.makedirs(os.path.dirname(SCREENSHOT_PATH), exist_ok=True)
        cv2.imwrite(SCREENSHOT_PATH, frame)
        print(f"[INFO] Screenshot saved: {os.path.abspath(SCREENSHOT_PATH)}")
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print(f"\n[INFO] Session ended. Violation log saved to: {logger.csv_path}")