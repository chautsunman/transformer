import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

neg_inf = 1e-9

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        # An attention function can be described as mapping a query and a set of key-value pairs to an output,
        # where the query, keys, values, and output are all vectors.

        # q: query, (batch_size, input_seq_lenth, d_k) or (batch_size, num_heads, input_seq_lenth, d_k)
        # k: key, (batch_size, input_seq_lenth, d_k) or (batch_size, num_heads, input_seq_lenth, d_k)
        # v: value, (batch_size, input_seq_lenth, d_v) or (batch_size, num_heads, input_seq_lenth, d_v)
        # mask: (batch_size, input_seq_length)
        # output: (batch_size, input_seq_lenth, d_v) or (batch_size, num_heads, input_seq_lenth, d_v)

        output = torch.matmul(q, k.transpose())
        # output: (batch_size, input_seq_lenth, input_seq_length) or (batch_size, num_heads, input_seq_lenth, input_seq_length)
        d_k = q.size(dim=0)
        output = output / math.sqrt(d_k)
        # mask output to not attend (True --> attend, False --> not attend)
        if attention_mask is not None:
            output = output.masked_fill(~attention_mask, neg_inf)
        output = F.softmax(output)
        attention_matrix = output
        output = torch.matmul(output, v)
        # output: (batch_size, input_seq_lenth, d_v) or (batch_size, num_heads, input_seq_lenth, d_v)
        return output
