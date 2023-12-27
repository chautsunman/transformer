import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, target_vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_encoders: int, num_decoders: int):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_prob=dropout_prob,
            num_encoders=num_encoders
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_prob=dropout_prob,
            num_decoders=num_decoders
        )
        self.final_layer = nn.Linear(target_vocab_size)

    def forward(self, context, x):
        encoder_output = self.encoder(context)
        # (batch_size, seq_length, d_model)

        decoder_output = self.decoder(x, encoder_output)
        # (batch_size, seq_length, d_model)

        y = self.final_layer(decoder_output)
        # (batch_size, seq_length, target_vocab_size)

        return y
