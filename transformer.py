import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    @staticmethod
    def get_default_transformer(vocab_size: int, target_vocab_size: int):
        # values from original paper
        return Transformer(
            vocab_size=vocab_size,
            target_vocab_size=target_vocab_size,
            d_model=512,
            num_heads=8,
            d_ff=2048,
            dropout_prob=0.1,
            num_encoders=6,
            num_decoders=6
        )

    def __init__(self, vocab_size: int, target_vocab_size: int, d_model: int, num_heads: int, d_ff: int, dropout_prob: float, num_encoders: int, num_decoders: int):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_prob=dropout_prob,
            num_encoders=num_encoders,
            input_seq_length=2048
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_prob=dropout_prob,
            num_decoders=num_decoders,
            input_seq_length=2048
        )
        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self, context, x):
        # context: (batch_size, seq_length), e.g. lang1 (machine translation)
        # x: (batch_size, seq_length), e.g. lang2 (machine translation)

        encoder_output = self.encoder(context)
        # (batch_size, seq_length, d_model)

        decoder_output = self.decoder(x, encoder_output)
        # (batch_size, seq_length, d_model)

        y = self.final_layer(decoder_output)
        # (batch_size, seq_length, target_vocab_size)

        return y
