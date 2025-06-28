import cv2

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class"]} {det["confidence"]:.2f}'

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Store previous positions for tracked objects
track_history = {}

def draw_trails(frame, tracked_objects):
    for obj in tracked_objects:
        if obj["class"] != "id 1":
            continue

        x1, y1, x2, y2 = obj["bbox"]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.arrowedLine(frame, (center_x, center_y + 20), (center_x, center_y - 20),
                        (255, 0, 255), 2, tipLength=0.5)
    return frame