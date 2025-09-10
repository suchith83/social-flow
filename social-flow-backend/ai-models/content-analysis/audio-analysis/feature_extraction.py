"""
Feature Extraction Module

Extracts:
- MFCC
- Spectral features
- Pretrained embeddings (wav2vec2, HuBERT, etc.)
"""

import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from .config import AudioConfig


class FeatureExtractor:
    def __init__(self, sample_rate=AudioConfig.SAMPLE_RATE, n_mfcc=AudioConfig.N_MFCC):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        # Load Hugging Face pretrained Wav2Vec2
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        return mfcc.T

    def extract_spectral(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral centroid and rolloff"""
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate).T
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate).T
        return np.hstack([centroid, rolloff])

    def extract_wav2vec2(self, y: np.ndarray) -> np.ndarray:
        """Extract embeddings from Wav2Vec2"""
        inputs = self.processor(y, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def extract_all(self, y: np.ndarray) -> dict:
        """Extract all features"""
        return {
            "mfcc": self.extract_mfcc(y),
            "spectral": self.extract_spectral(y),
            "wav2vec2": self.extract_wav2vec2(y),
        }
