import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder_layer import EncoderLayer
from position_encoding import PositionEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_encoders: int):
        super().__init__()

        # num_encoders = 6
        self.num_encoders = num_encoders

        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding_layer = PositionEncoding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(f"encoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_encoders)])

    def forward(self, x):
        # x: (batch_size, input_seq_length)
        y = x

        y = self.embedding_layer(y)
        y = y + self.position_embedding_layer(y)
        # (batch_size, input_seq_length, d_model)

        for layer in self.encoder_layers:
            y = layer(y)
        # (batch_size, input_seq_length, d_model)

        return y
