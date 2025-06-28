import cv2
import numpy as np

class PathVisualizer:
    def __init__(self, max_points=50):
        self.points = []
        self.max_points = max_points

    def update_path(self, point):
        self.points.append(point)
        if len(self.points) > self.max_points:
            self.points.pop(0)

    def draw_path(self, frame):
        for i in range(1, len(self.points)):
            pt1 = self.points[i - 1]
            pt2 = self.points[i]
            if pt1 is None or pt2 is None:
                continue
            alpha = i / len(self.points)
            color = (255 * (1 - alpha), 0, 255 * alpha)  # fade from pink to blue
            cv2.line(frame, pt1, pt2, color, 4)
        return frame
