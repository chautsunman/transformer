import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multi_head_attention import MultiHeadAttention
from position_wise_feed_foward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, name: str, d_model: int, num_heads: int, d_ff: int, dropout_prob: float):
        super().__init__()

        self.name = name
        # d_model = 512
        self.d_model = d_model
        # num_heads = 8
        self.num_heads = num_heads
        # d_ff = 2048
        self.d_ff = d_ff
        # dropout_prob = 0.1
        self.dropout_prob = dropout_prob

        self.decoder_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_multi_head_attention_dropout = nn.Dropout(dropout_prob)
        self.decoder_multi_head_attention_layer_norm = nn.LayerNorm(d_model)

        self.encoder_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_multi_head_attention_dropout = nn.Dropout(dropout_prob)
        self.encoder_multi_head_attention_layer_norm = nn.LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.ff_dropout = nn.Dropout(dropout_prob)
        self.ff_layer_norm = nn.LayerNorm(d_model)

    def forward(self, decoder_input, encoder_output, decoder_mask):
        y = self._forward_decoder_attention(decoder_input, decoder_mask)
        y = self._forward_encoder_attention(decoder_input, encoder_output)
        y = self._forward_ff(y)
        return y

    def _forward_decoder_attention(self, decoder_input, decoder_mask):
        y = self.decoder_multi_head_attention(decoder_input, decoder_input, decoder_input, mask=decoder_mask)
        y = self.decoder_multi_head_attention_dropout(y)
        y = decoder_input + y
        y = self.decoder_multi_head_attention_layer_norm(y)
        return y

    def _forward_encoder_attention(self, decoder_input, encoder_output):
        # query = encoder_output
        y = self.encoder_multi_head_attention(encoder_output, decoder_input, decoder_input)
        y = self.encoder_multi_head_attention_dropout(y)
        y = decoder_input + y
        y = self.encoder_multi_head_attention_layer_norm(y)
        return y

    def _forward_ff(self, x):
        y = self.ff(x)
        y = self.ff_dropout(y)
        y = x + y
        y = self.ff_layer_norm(y)
        return y
