"""
High-level caption generation wrapper.
"""

import torch
from .decoder import greedy_decode, beam_search_decode


class CaptionGenerator:
    def __init__(self, encoder, decoder, vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

    def generate(self, image, method="beam", beam_size=5):
        if method == "greedy":
            return greedy_decode(self.encoder, self.decoder, image, self.vocab)
        else:
            return beam_search_decode(self.encoder, self.decoder, image, self.vocab, beam_size=beam_size)
