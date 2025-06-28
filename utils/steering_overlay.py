import cv2
import numpy as np

def overlay_steering_wheel(frame, angle):
    steering_img = cv2.imread("assets/images/steering_wheel.png", cv2.IMREAD_UNCHANGED)
    if steering_img is None or steering_img.shape[2] != 4:
        print("Error loading PNG with alpha.")
        return frame

    # Resize the wheel (e.g. 100×100)
    wheel = cv2.resize(steering_img, (100, 100), interpolation=cv2.INTER_AREA)
    h, w = wheel.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(wheel, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Separate color and alpha
    overlay_bgr = rotated[:, :, :3]
    mask = rotated[:, :, 3] / 255.0

    # Position: bottom right corner with 10px margin
    y1, y2 = frame.shape[0] - h - 10, frame.shape[0] - 10
    x1, x2 = frame.shape[1] - w - 10, frame.shape[1] - 10

    # Overlay using mask
    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            mask * overlay_bgr[:, :, c] +
            (1 - mask) * frame[y1:y2, x1:x2, c]
        )
    return frame
