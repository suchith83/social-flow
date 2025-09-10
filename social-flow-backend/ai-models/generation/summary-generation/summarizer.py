"""
High-level summarizer wrapper exposing:
- extractive_summarize(document) -> list of sentences (indices or text)
- abstractive_summarize(document) -> generated summary string

This component orchestrates preprocessing, encoding, decoding.
"""

from typing import List
import torch
from .decoder import beam_search_decode, greedy_decode, top_k_top_p_sampling
from .utils import logger

class Summarizer:
    def __init__(self, abstractive_model=None, abstractive_tokenizer=None,
                 extractive_model=None, sentence_splitter=None, sentence_encoder=None, device="cuda"):
        """
        Provide at least one of abstractive_model or extractive_model.
        sentence_splitter: function that turns document text -> list of sentences
        sentence_encoder: function that turns sentence -> vector tensor for extractive model
        """
        self.abstractive_model = abstractive_model
        self.abstractive_tokenizer = abstractive_tokenizer
        self.extractive_model = extractive_model
        self.sentence_splitter = sentence_splitter
        self.sentence_encoder = sentence_encoder
        self.device = device

    # -------------------------
    # Extractive summarization
    # -------------------------
    def extractive_summarize(self, document: str, top_k: int = 3) -> List[int]:
        if self.extractive_model is None:
            raise RuntimeError("Extractive model not provided")
        sents = self.sentence_splitter(document)
        encs = [self.sentence_encoder(s).unsqueeze(0) for s in sents]
        sent_feats = torch.cat(encs, dim=0).unsqueeze(0)  # (1, num_sents, feat_dim)
        with torch.no_grad():
            scores = self.extractive_model(sent_feats).squeeze(0).cpu().numpy()  # (num_sents,)
        top_idx = scores.argsort()[::-1][:top_k]
        ordered = sorted(top_idx.tolist())
        summary_sents = [sents[i] for i in ordered]
        return summary_sents

    # -------------------------
    # Abstractive summarization
    # -------------------------
    def abstractive_summarize(self, document: str, method: str = "beam", beam_size: int = 4, max_len: int = None) -> str:
        if self.abstractive_model is None or self.abstractive_tokenizer is None:
            raise RuntimeError("Abstractive model or tokenizer not provided")
        max_len = max_len or 200
        input_ids = torch.tensor([self.abstractive_tokenizer.encode(document, max_length=1024)], dtype=torch.long)
        if method == "greedy":
            token_ids = greedy_decode(self.abstractive_model, input_ids, self.abstractive_tokenizer, max_len)
        elif method == "sampling":
            token_ids = top_k_top_p_sampling(self.abstractive_model, input_ids, self.abstractive_tokenizer, max_len)
        else:
            token_ids = beam_search_decode(self.abstractive_model, input_ids, self.abstractive_tokenizer, beam_size=beam_size, max_len=max_len)
        # decode to text
        text = self.abstractive_tokenizer.decode(token_ids[1:])  # remove SOS
        return text
