import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from decoder_layer import DecoderLayer
from position_encoding import PositionEncoding

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_decoders: int):
        super().__init__()

        # num_decoders = 6
        self.num_decoders = num_decoders

        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.position_embedding_layer = PositionEncoding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([DecoderLayer(f"decoder_{i}", d_model, num_heads, d_ff, dropout_prob) for i in range(num_decoders)])

    def forward(self, decoder_input, encoder_output, decoder_mask):
        # x: (batch_size, output_seq_length)
        y = decoder_input

        y = self.embedding_layer(y)
        y = y + self.position_embedding_layer(y)
        # (batch_size, input_seq_length, d_model)

        for layer in self.decoder_layers:
            y = layer(y, encoder_output, decoder_mask)
        # (batch_size, input_seq_length, d_model)

        return y
