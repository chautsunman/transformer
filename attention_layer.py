import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from multi_head_attention import MultiHeadAttention

class BaseAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_prob: float):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.multi_head_attention(x, x, x)
        y = self.multi_head_attention_dropout(y)
        y = x + y
        y = self.multi_head_attention_layer_norm(y)
        return y

class GlobalSelfAttention(BaseAttention):
    def forward(self, x):
        y = self.multi_head_attention(x, x, x)
        y = self.dropout(y)
        y = x + y
        y = self.layer_norm(y)
        return y

class CrossAttention(BaseAttention):
    def forward(self, x, context):
        y = self.multi_head_attention(x, context, context)
        y = self.dropout(y)
        y = x + y
        y = self.layer_norm(y)
        return y

class CausalSelfAttention(BaseAttention):
    def forward(self, x):
        y = self.multi_head_attention(x, x, x, True)
        y = self.dropout(y)
        y = x + y
        y = self.layer_norm(y)
        return y
