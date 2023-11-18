# coding=utf-8

import torch
from torch import nn


class SMU(nn.Module):

    def __init__(self, alpha=0.01, mu=2.5):
        super(SMU, self).__init__()
        self.alpha = alpha
        # initialize mu
        self.mu = torch.nn.Parameter(torch.tensor(mu))

    def forward(self, x):
        return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(
            self.mu * (1 - self.alpha) * x)) / 2
