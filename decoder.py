import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder_layer import DecoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_decoders: int):
        super().__init__()

        # num_decoders = 6
        self.num_decoders = num_decoders

        self.decoder_layers = nn.ModuleList([DecoderLayer(f"decoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_decoders)])

    def forward(self, decoder_input, encoder_output, decoder_mask):
        y = decoder_input
        for layer in self.decoder_layers:
            y = layer(y, encoder_output, decoder_mask)
        return y
