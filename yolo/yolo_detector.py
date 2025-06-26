import torch
from ultralytics import YOLO
import cv2

class YoloDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def detect(self, frame):
        # Convert BGR to RGB as YOLO expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = self.model.predict(rgb_frame, verbose=False)

        # Extract detection results
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": name
                })

        return detections
