import cv2
from yolo.yolo_detector import YoloDetector
from utils.draw import draw_detections

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 = webcam; or replace with 'data/test_video.mp4'

    # Load YOLO model
    detector = YoloDetector(weights_path='yolo11n.pt')


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        detections = detector.detect(frame)

        # Draw the results on the frame
        frame = draw_detections(frame, detections)

        # Show the frame
        cv2.imshow("Self-Driving View", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
