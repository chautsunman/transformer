import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_encoders: int):
        super().__init__()

        # num_encoders = 6
        self.num_encoders = num_encoders

        self.encoder_layers = nn.ModuleList([EncoderLayer(f"encoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_encoders)])

    def forward(self, x):
        y = x
        for layer in self.encoder_layers:
            y = layer(y)
        return y
