import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.d_model = d_model  # original paper = 512
        self.d_ff = d_ff  # original paper = 2048

        self.linear_1_func = nn.Linear(d_model, d_ff)
        self.relu_func = nn.ReLU()
        self.linear_2_func = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, d_model)
        # y: (batch_size, d_ff)
        y = self.linear_1_func(x)
        y = self.relu_func(y)
        # y: (batch_size, d_model)
        y = self.linear_2_func(x)
        return y
