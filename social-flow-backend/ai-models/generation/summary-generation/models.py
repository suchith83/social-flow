"""
Model implementations for abstractive and extractive summarization.

Abstractive: Transformer Seq2Seq implemented in PyTorch (Encoder & Decoder).
Extractive: Simple sentence scorer network (Bi-LSTM + attention) to rank sentences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import HIDDEN_SIZE, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_HEADS, FF_DIM, DROPOUT, VOCAB_SIZE
from typing import Optional

# -----------------------
# Abstractive seq2seq
# -----------------------
class TransformerSeq2Seq(nn.Module):
    """
    Full Transformer seq2seq for abstractive summarization.

    Note:
      - This implementation uses PyTorch's nn.Transformer as building block.
      - Input and target are token ids. Embeddings + positional encodings are included.
      - For production-scale performance, prefer HuggingFace models (Bart/T5) — code here is educational and usable.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = HIDDEN_SIZE,
                 num_encoder_layers: int = NUM_ENCODER_LAYERS, num_decoder_layers: int = NUM_DECODER_LAYERS,
                 nhead: int = NUM_HEADS, dim_feedforward: int = FF_DIM, dropout: float = DROPOUT,
                 max_seq_length: int = 2048, pad_idx: int = 0):
        super().__init__()
        self.model_type = "transformer_seq2seq"
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        self.src_tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len=max_seq_length)

        self.transformer = nn.Transformer(d_model=embed_dim,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)  # batch_first for convenience

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def generate_src_mask(self, src, src_pad_idx):
        # mask where src == pad
        return (src == src_pad_idx)

    def forward(self, src_ids, tgt_ids):
        """
        src_ids: (batch, src_len)
        tgt_ids: (batch, tgt_len)
        returns logits over vocabulary (batch, tgt_len, vocab_size)
        """
        src_mask = (src_ids == self.pad_idx)  # (batch, src_len)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)  # (tgt_len, tgt_len)

        # embeddings + positional
        src_emb = self.positional_encoding(self.src_tok_emb(src_ids))  # (batch, src_len, embed)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_ids))

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        logits = self.fc_out(outs)
        return logits

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (or learned via nn.Parameter if preferred).
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # create positional encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -----------------------
# Extractive model
# -----------------------
class SentenceScorer(nn.Module):
    """
    Score each sentence for extractive summarization.
    Uses a Bi-LSTM over sentence embeddings plus self-attention to compute scores.
    Input shape: (batch, num_sentences, sent_feat_dim)
    Output: (batch, num_sentences) scores
    """

    def __init__(self, sent_feat_dim=768, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.sent_feat_dim = sent_feat_dim
        self.hidden_dim = hidden_dim
        self.bi_lstm = nn.LSTM(input_size=sent_feat_dim, hidden_size=hidden_dim // 2,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sent_feats, mask=None):
        """
        sent_feats: (batch, num_sentences, sent_feat_dim)
        mask: optional (batch, num_sentences) boolean mask where True indicates padding (ignored)
        """
        out, _ = self.bi_lstm(sent_feats)  # (batch, num_sentences, hidden_dim)
        attn_logits = self.attn(out).squeeze(-1)  # (batch, num_sentences)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.bool(), float("-inf"))
        scores = torch.sigmoid(attn_logits)  # per-sentence probability [0,1]
        return scores
