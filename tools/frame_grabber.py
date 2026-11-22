import cv2

VIDEO_PATH = "media/videos/traffic.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
if success:
    cv2.imwrite("traffic_sample_frame.jpg", frame)
    print("Saved as traffic_sample_frame.jpg")
cap.release()