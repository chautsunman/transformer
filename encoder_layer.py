import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multi_head_attention import MultiHeadAttention
from position_wise_feed_foward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float):
        super().__init__()

        # d_model = 512
        self.d_model = d_model
        # num_heads = 8
        self.num_heads = num_heads
        # d_ff = 2048
        self.d_ff = d_ff
        # dropout_prob = 0.1
        self.dropout_prob = dropout_prob

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention_dropout = nn.Dropout(dropout_prob)
        self.multi_head_attention_layer_norm = nn.LayerNorm(d_model)

        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.ff_dropout = nn.Dropout(dropout_prob)
        self.ff_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self._forward_attention(x)
        y = self._forward_ff(y)
        return y

    def _forward_attention(self, x):
        y = self.multi_head_attention(x, x, x)
        y = self.multi_head_attention_dropout(y)
        y = x + y
        y = self.multi_head_attention_layer_norm(y)
        return y

    def _forward_ff(self, x):
        y = self.ff(x)
        y = self.ff_dropout(y)
        y = x + y
        y = self.ff_layer_norm(y)
        return y
