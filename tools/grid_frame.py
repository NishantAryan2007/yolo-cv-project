import cv2

VIDEO_PATH = "media/videos/footpath.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()

if success:
    h, w = frame.shape[:2]
    print("Frame size (width, height):", w, h)
    for y in range(0, h, 100):
        cv2.line(frame, (0, y), (w, y), (0, 0, 255), 1)
        cv2.putText(frame, str(y), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite("grid_frame.jpg", frame)
    print("Saved grid_frame.jpg")
else:
    print("Could not read video — check the path/filename")

cap.release()