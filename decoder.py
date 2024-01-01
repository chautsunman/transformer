import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder_layer import DecoderLayer
from position_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_decoders: int, input_seq_length: int):
        super().__init__()

        self.num_decoders = num_decoders  # original paper = 6

        self.positional_embedding = PositionalEncoding(vocab_size, input_seq_length, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.decoder_layers = nn.ModuleList([DecoderLayer(f"decoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_decoders)])

    def forward(self, decoder_input, encoder_output, decoder_mask):
        # x: (batch_size, output_seq_length)
        y = decoder_input

        y = self.positional_embedding(y)
        # (batch_size, input_seq_length, d_model)

        y = self.dropout(y)

        for layer in self.decoder_layers:
            y = layer(y, encoder_output, decoder_mask)
        # (batch_size, input_seq_length, d_model)

        return y
