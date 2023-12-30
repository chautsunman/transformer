import torch

def get_causal_mask(q):
    # usage: attention mask (True --> attend, False --> not attend)
    # e.g. q: (batch_size, num_heads, input_seq_length=4, d_k)
    # output:
    # [[[[True, False, False, False],
    #    [True, True,  False, False],
    #    [True, True,  True,  False],
    #    [True, True,  True,  True]]]]
    ones_mask = torch.ones(1, 1, q.size(dim=2), q.size(dim=2))
    row_idxs = torch.cumsum(ones_mask, dim=-2)
    col_idxs = torch.cumsum(ones_mask, dim=-1)
    return row_idxs > col_idxs
