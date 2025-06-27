from utils.sort import Sort
import numpy as np

# Object tracking using the SORT algorithm
class ObjectTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(detections)
            if dets.ndim == 1:
                dets = np.expand_dims(dets, axis=0)
        
        tracked = self.tracker.update(dets)
        return tracked
