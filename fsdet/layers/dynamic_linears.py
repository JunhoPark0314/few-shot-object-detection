from collections import Iterable
import itertools

import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import nn


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(nn.Linear(c_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, nof_kernels))

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)

class AttentionLayerWoGP(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.to_scores = nn.Sequential(nn.Linear(c_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, nof_kernels))

    def forward(self, x, batch_size, temperature=1):
        cond_batch_size = x.shape[0]
        scores = self.to_scores(x)
        scores = scores.view(1, cond_batch_size, -1).repeat(batch_size, 1, 1)
        return F.softmax(scores / temperature, dim=-1)

class DynamicCondLinear(TempModule):
    def __init__(self, nof_kernels, reduce, in_channels, out_channels, init_bias=None, bias=True):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.nof_kernels = nof_kernels
        self.attention = AttentionLayerWoGP(in_channels, max(1, in_channels // reduce), nof_kernels)
        self.kernels_weights = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels), requires_grad=True)
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)
        self.initialize_parameters()
        if init_bias is not None:
            assert bias is True
            bias_value = -(math.log((1 - init_bias) / init_bias))
            torch.nn.init.constant_(self.kernels_bias, bias_value)

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            nn.init.normal_(self.kernels_weights[i_kernel], std=0.01)
        if self.kernels_bias is not None:
            nn.init.constant_(self.kernels_bias, 0)

    def forward(self, x, condition, temperature=1):
        batch_size = x.shape[0]

        alphas = self.attention(condition, batch_size, temperature)
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.permute(0, 1, 2).unsqueeze(0), alphas.permute(0, 2, 1).unsqueeze(-1)), dim=1)

        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.permute(0, 2, 1)), dim=1)
        else:
            agg_bias = None

        out = torch.sum(torch.mul(agg_weights.permute(0, 2, 1), x.unsqueeze(-1)), dim=1) + agg_bias

        return out
