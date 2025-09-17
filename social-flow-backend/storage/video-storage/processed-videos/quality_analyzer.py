"""
Simple quality analysis (proxy for VMAF/SSIM)
"""

import random


class QualityAnalyzer:
    def analyze(self, input_file: str) -> dict:
        # Mock quality metrics (real systems would run ffmpeg VMAF filters)
        return {
            "vmaf": round(random.uniform(80, 95), 2),
            "ssim": round(random.uniform(0.9, 1.0), 3),
            "psnr": round(random.uniform(30, 45), 2),
        }
