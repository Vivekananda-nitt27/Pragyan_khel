import cv2
import numpy as np
from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.selected = None
        self.click_point = None

    def select(self, x, y):
        self.click_point = (x, y)

    def process(self, frame):

        results = self.model(frame)[0]
        detections = []

        # ===== DETECTIONS =====
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append((cls, (x1, y1, x2, y2), (cx, cy)))

        # ===== SELECTION =====
        if self.click_point is not None:
            px, py = self.click_point

            for cls, box, center in detections:
                x1, y1, x2, y2 = box
                if x1 <= px <= x2 and y1 <= py <= y2:
                    self.selected = {"cls": cls, "center": center}
                    break

            self.click_point = None

        # ===== TRACKING =====
        tracked_box = None

        if self.selected is not None:
            best = None
            best_dist = 1e9

            for cls, box, center in detections:
                if cls != self.selected["cls"]:
                    continue

                dist = np.linalg.norm(
                    np.array(center) - np.array(self.selected["center"])
                )

                if dist < best_dist:
                    best_dist = dist
                    best = (box, center)

            if best:
                tracked_box, center = best
                self.selected["center"] = center

        # ⭐ IMPORTANT CHANGE
        # If nothing selected → return original frame
        if self.selected is None:
            return frame

        # ===== FOCUS BLUR =====
        output = cv2.GaussianBlur(frame, (91, 91), 0)

        if tracked_box is not None:
            x1, y1, x2, y2 = tracked_box

            output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output