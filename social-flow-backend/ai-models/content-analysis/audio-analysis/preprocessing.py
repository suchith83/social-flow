"""
Audio Preprocessing Module

Handles:
- Loading audio
- Resampling
- Noise reduction
- Silence trimming
- Normalization
"""

import librosa
import numpy as np
import noisereduce as nr
from .config import AudioConfig


class AudioPreprocessor:
    def __init__(self, sample_rate: int = AudioConfig.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load(self, filepath: str) -> np.ndarray:
        """Load an audio file"""
        y, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        return y

    def trim_silence(self, y: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence"""
        yt, _ = librosa.effects.trim(y, top_db=20)
        return yt

    def reduce_noise(self, y: np.ndarray) -> np.ndarray:
        """Apply spectral gating noise reduction"""
        return nr.reduce_noise(y=y, sr=self.sample_rate)

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize waveform between -1 and 1"""
        return librosa.util.normalize(y)

    def preprocess(self, filepath: str) -> np.ndarray:
        """Full preprocessing pipeline"""
        y = self.load(filepath)
        y = self.trim_silence(y)
        y = self.reduce_noise(y)
        y = self.normalize(y)
        return y
