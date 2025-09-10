"""
Video Preprocessing Module

Extracts frames at given FPS, resizes & normalizes them.
"""

import cv2
from torchvision import transforms
from PIL import Image
import os
from .config import SceneConfig


class VideoPreprocessor:
    def __init__(self, frame_rate=SceneConfig.FRAME_RATE, img_size=SceneConfig.IMG_SIZE):
        self.frame_rate = frame_rate
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path: str, save_dir: str = None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        step = max(1, fps // self.frame_rate)

        frames = []
        idx = 0
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = self.transform(img)
                frames.append(tensor)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    img.save(os.path.join(save_dir, f"frame_{frame_id:05d}.jpg"))
                frame_id += 1
            idx += 1

        cap.release()
        return frames
