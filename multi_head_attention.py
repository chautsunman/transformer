import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model  # original paper = 512
        self.num_heads = num_heads  # original paper = 8
        self.d_k = d_model / num_heads  # original paper = dmodel/h = 64
        self.d_v = d_model / num_heads  # original paper = dmodel/h = 64

        # 512 --> 512 linear projection
        # combined all 8 heads weight matrixes, separate into multiple heads during calculation
        self.q_linear_projection_func = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.k_linear_projection_func = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.v_linear_projection_func = nn.Linear(d_model, num_heads * self.d_v, bias=False)

        # 512 --> 512 linear projection
        self.attention_projection_func = nn.Linear(num_heads * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # original paper:
        # Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
        # we found it beneficial to linearly project the queries, keys and values h times with different, learned
        # linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of
        # queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional
        # output values.

        # q: query, (batch_size, input_seq_lenth, d_k)
        # k: key, (batch_size, input_seq_lenth, d_k)
        # v: value, (batch_size, input_seq_lenth, d_v)
        # mask: (batch_size, input_seq_length)
        # output: (batch_size, input_seq_lenth, d_v)

        batch_size = q.size(0)

        # project q, k and v
        projected_q = self.q_linear_projection_func(q)
        projected_k = self.k_linear_projection_func(k)
        projected_v = self.v_linear_projection_func(v)
        # re-organize q, k and v tensors into multiple heads
        # projected_q: (batch_size, input_seq_lenth, num_heads, d_k) (e.g. (1, 512, 8, 64))
        # projected_k: (batch_size, input_seq_lenth, num_heads, d_k) (e.g. (1, 512, 8, 64))
        # projected_v: (batch_size, input_seq_lenth, num_heads, d_v) (e.g. (1, 512, 8, 64))
        projected_q = projected_q.view(batch_size, q.size(1), self.num_heads, self.d_k)
        projected_k = projected_k.view(batch_size, k.size(1), self.num_heads, self.d_k)
        projected_v = projected_v.view(batch_size, v.size(1), self.num_heads, self.d_v)
        # projected_q: (batch_size, num_heads, input_seq_lenth, d_k) (e.g. (1, 8, 512, 64))
        # projected_k: (batch_size, num_heads, input_seq_lenth, d_k) (e.g. (1, 8, 512, 64))
        # projected_v: (batch_size, num_heads, input_seq_lenth, d_v) (e.g. (1, 8, 512, 64))
        projected_q = projected_q.transpose(1, 2)
        projected_k = projected_k.transpose(1, 2)
        projected_v = projected_v.transpose(1, 2)

        # attention: (batch_size, num_heads, input_seq_length, d_v)
        attention = self.attention(projected_q, projected_k, projected_v, mask=mask)

        # swap attention dimensions
        # attention: (batch_size, input_seq_length, num_heads, d_v)
        attention = attention.transpose(1, 2)

        # concat attention values across heads
        # attention: (batch_size, input_seq_length, d_model)
        attention = attention.contiguous().view(batch_size, q.size(1), self.d_model)

        # project attention
        # attention: (batch_size, input_seq_lenth, d_v)
        attention_projection = self.attention_projection_func(attention)
        multi_head_attention = attention_projection

        return multi_head_attention
