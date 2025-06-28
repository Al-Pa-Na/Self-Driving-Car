import cv2
import time

from yolo.yolo_detector import YoloDetector
from utils.draw import draw_detections
from utils.lane_detection import detect_lane, get_lane_direction
from utils.tracker import ObjectTracker
from utils.controller import decide_steering_action
from utils.proximity import check_proximity
from utils.steering_overlay import overlay_steering_wheel
from utils.path_visualizer import PathVisualizer
from utils.draw import draw_trails
from collections import deque, Counter

direction_history = deque(maxlen = 8)

def main():
    use_webcam = False

    video_path = 'assets/videos/test_video1.mp4'
    cap = cv2.VideoCapture(0 if use_webcam else video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))


    # Load YOLO model
    detector = YoloDetector(weights_path='yolo11n.pt')
    
    tracker = ObjectTracker()
    path_viz = PathVisualizer()

    prev_time = 0
    current_angle = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        # Run YOLO detection
        detections = detector.detect(frame)
        
        # Convert YOLO detections to SORT format: [x1, y1, x2, y2, conf]
        dets_for_sort = [
            [*det["bbox"], det["confidence"]]
            for det in detections
            if "bbox" in det and "confidence" in det
            ]
        
        # Track detected objects across frames
        tracked_objects_np = tracker.update(dets_for_sort)
        tracked_objects = []
        for obj in tracked_objects_np:
            x1, y1, x2, y2, track_id = obj.astype(int)
                
            tracked_objects.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": 1.0,  # Or track confidence if available
            "class": f"id {track_id}"
            })

        # Identify the object closest to bottom (likely own car)
        own_car = min(tracked_objects, key=lambda o: o['bbox'][1]) if tracked_objects else None
        if own_car:
            bbox = own_car["bbox"]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            path_viz.update_path((center_x, center_y))
            
        # Draw tracked detections on the frame
        frame = draw_detections(frame, tracked_objects)
        
        frame = draw_trails(frame, tracked_objects)
        
        # Apply lane detection and get frame with lines info
        frame, lines = detect_lane(frame)

        # Get current lane direction
        current_direction = get_lane_direction(lines)
        
        # Append to history and compute the most frequent direction
        direction_history.append(current_direction)
        direction = Counter(direction_history).most_common(1)[0][0]

        # Decide steering action based on direction
        action = decide_steering_action(direction)
        
        target_angle = action
            
        # Smooth transition
        steering_speed = 5  # degrees per frame
        if current_angle < target_angle:
            current_angle = min(current_angle + steering_speed, target_angle)
        elif current_angle > target_angle:
            current_angle = max(current_angle - steering_speed, target_angle)
        
        # Get proximity status
        frame_height, frame_width = frame.shape[:2]
        proximity_status = check_proximity(tracked_objects, frame_width, frame_height)

        # Show proximity status
        cv2.putText(frame, proximity_status, (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255) if proximity_status != "Clear" else (0, 255, 0), 2)

        # Display steering action on the frame
        cv2.putText(frame, f"Action: {action}", (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        frame = overlay_steering_wheel(frame, current_angle)

        
        #Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        #Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame = path_viz.draw_path(frame)
        
        out.write(frame)
        # Show the frame
        cv2.imshow("Self-Driving View", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
