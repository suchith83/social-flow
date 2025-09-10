"""
Models: CNN Encoder + Transformer Decoder for caption generation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .config import EMBED_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, dropout=DROPOUT, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        memory = features.unsqueeze(1)
        out = self.transformer_decoder(embeddings.transpose(0, 1), memory.transpose(0, 1), tgt_mask=tgt_mask)
        out = self.fc_out(out.transpose(0, 1))
        return out
