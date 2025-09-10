"""
Scene Detection using Histogram + Feature Differences
"""

import cv2
import numpy as np
from .config import SceneConfig


class SceneDetector:
    def __init__(self, threshold=SceneConfig.HIST_DIFF_THRESHOLD):
        self.threshold = threshold

    def detect_scenes(self, video_path: str):
        """Detect scene boundaries using histogram differences"""
        cap = cv2.VideoCapture(video_path)
        prev_hist = None
        scenes = [0]
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > self.threshold:
                    scenes.append(frame_idx)
            prev_hist = hist
            frame_idx += 1

        cap.release()
        return scenes
