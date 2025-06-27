import cv2
import time

from yolo.yolo_detector import YoloDetector
from utils.draw import draw_detections
from utils.lane_detection import detect_lane, get_lane_direction
from utils.tracker import ObjectTracker

def main():
    use_webcam = False

    video_path = 'assets/videos/test_video.mp4'
    cap = cv2.VideoCapture(0 if use_webcam else video_path)

    # Load YOLO model
    detector = YoloDetector(weights_path='yolo11n.pt')
    
    tracker = ObjectTracker()

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Draw tracked detections on the frame
        frame = draw_detections(frame, tracked_objects)
        
        # Apply lane detection and get frame with lines info
        frame, lines = detect_lane(frame)

        # Get lane direction based on detected lines
        direction = get_lane_direction(lines)

        # Display lane direction on the frame
        cv2.putText(frame, direction, (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        
        #Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        #Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Self-Driving View", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
