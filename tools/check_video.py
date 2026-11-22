import cv2

VIDEO_PATH = "media/videos/traffic.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {width}x{height}")
cap.release()