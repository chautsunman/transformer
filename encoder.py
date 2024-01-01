import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder_layer import EncoderLayer
from position_encoding import PositionalEmbedding

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_encoders: int, input_seq_length: int):
        super().__init__()

        self.num_encoders = num_encoders  # original paper = 6

        self.positional_embedding = PositionalEmbedding(vocab_size, input_seq_length, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.encoder_layers = nn.ModuleList([EncoderLayer(f"encoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_encoders)])

    def forward(self, x):
        # x: (batch_size, input_seq_length), each value is a token ID
        y = x

        y = self.positional_embedding(y)
        # (batch_size, input_seq_length, d_model)

        y = self.dropout(y)

        for layer in self.encoder_layers:
            y = layer(y)
        # (batch_size, input_seq_length, d_model)

        return y
