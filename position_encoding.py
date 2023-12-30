import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionEncoding(nn.Module):
    def __init__(self, input_seq_length: int, d_embedding: int):
        super().__init__()

        self.input_seq_length = input_seq_length
        self.d_embedding = d_embedding  # original paper = 512

        self.position_encoding_matrix = torch.zeros((input_seq_length, d_embedding))

        # input_seq_arr: (input_seq_length, 1)
        # e.g. [[0], [1], ..., [input_seq_length - 1]]
        input_seq_arr = torch.arange(input_seq_length)
        input_seq_arr = input_seq_arr.reshape(-1, 1)

        # embedding_arr: (d_embedding, )
        # e.g. [0, 1, ..., d_embedding - 1]
        embedding_arr = torch.arange(d_embedding)
        embedding_arr = embedding_arr * 2 / d_embedding
        embedding_arr = torch.pow(10000, embedding_arr)

        # x_matrix = (input_seq_length, d_embedding)
        x_matrix = input_seq_arr / embedding_arr

        self.position_encoding_matrix[:, 0::2] = torch.sin(x_matrix)
        self.position_encoding_matrix[:, 1::2] = torch.cos(x_matrix)

    def forward(self, x):
        # x: (batch_size, input_seq_length, d_embedding)
        # output: (batch_size, input_seq_length, d_embedding)

        x_input_seq_length = x.shape[1]
        # position_encoding: (batch_size, input_seq_length, d_embedding)
        position_encoding = self.position_encoding_matrix[:x_input_seq_length, :]
        position_encoding = torch.unsqueeze(position_encoding, 0)

        y = x + position_encoding
        return y
