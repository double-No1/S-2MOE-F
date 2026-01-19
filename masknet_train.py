import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ParametricMaskNet(nn.Module):
    def __init__(self, num_segments, mask_dim, mode='softmax', dropout=0.1):

        super(ParametricMaskNet, self).__init__()
        self.num_segments = num_segments
        self.mask_dim = mask_dim
        self.mode = mode
        self.raw_mask = Parameter(torch.randn(num_segments, mask_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self):

        if self.mode == 'softmax':
            mask = F.softmax(self.raw_mask, dim=-1)
        elif self.mode == 'sigmoid':
            mask = torch.sigmoid(self.raw_mask)
        elif self.mode == 'relu':
            mask = F.relu(self.raw_mask)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        mask = self.dropout(mask)
        return mask
