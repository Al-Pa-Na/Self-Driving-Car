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
    
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
            if 20 < angle < 160:  # remove nearly vertical and flat lines
                filtered_lines.append([[x1, y1, x2, y2]])


    # Draw the lines on a blank image
    line_image = np.zeros_like(frame)
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 3)


    # Combine the original frame with the line image
    combined = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return combined, filtered_lines

# Determine lane direction based on slopes of detected lines
def get_lane_direction(lines):
    if lines is None:
        return "No lane detected"

    left_slopes = []
    right_slopes = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # avoid division by zero
            slope = (y2 - y1) / (x2 - x1)
            # Categorize slopes as left or right lanes based on sign and threshold
            if slope < -0.3:
                left_slopes.append(slope)
            elif slope > 0.3:
                right_slopes.append(slope)

    # If no slopes detected, assume going straight
    if len(left_slopes) == 0 and len(right_slopes) == 0:
        return "Straight"

    avg_left = np.mean(left_slopes) if left_slopes else 0
    avg_right = np.mean(right_slopes) if right_slopes else 0

    # Decide turn direction based on dominant slope
    if abs(avg_left) > abs(avg_right):
        return "Turn Left"
    elif abs(avg_right) > abs(avg_left):
        return "Turn Right"
    else:
        return "Straight"
