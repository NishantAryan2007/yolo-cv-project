import cv2

mask = cv2.imread("traffic_mask.png")
print("Mask size:", mask.shape[1], mask.shape[0])  # width, height

if mask.shape[1] != 1280 or mask.shape[0] != 720:
    mask_resized = cv2.resize(mask, (1280, 720))
    cv2.imwrite("traffic_mask.png", mask_resized)
    print("Resized and overwritten to match video dimensions.")
else:
    print("Size already correct, no changes needed.")