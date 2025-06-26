import cv2
import numpy as np

def detect_lane(frame):
    height, width = frame.shape[:2]
    
    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest (bottom 40% of the frame)
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (0, int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect straight lines using Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=50
    )

    # Draw the lines on a blank image
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Combine the original frame with the line image
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined