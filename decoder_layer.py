import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from attention_layer import CausalSelfAttention, CrossAttention
from feed_forward_layer import FeedForwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, name: str, d_model: int, num_heads: int, d_ff: int, dropout_prob: float):
        super().__init__()

        self.name = name
        self.d_model = d_model  # original paper = 512
        self.num_heads = num_heads  # original paper = 8
        self.d_ff = d_ff  # original paper = 2048
        self.dropout_prob = dropout_prob  # original paper = 0.1

        self.causal_self_attention = CausalSelfAttention(d_model, num_heads, dropout_prob)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout_prob)
        self.ff = FeedForwardLayer(d_model, d_ff, dropout_prob)

    def forward(self, decoder_input, encoder_output):
        y = self.causal_self_attention(decoder_input)
        y = self.cross_attention(y, encoder_output)
        y = self.ff(y)
        return y
