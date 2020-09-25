
import torch
import torch.nn as nn
import numpy as np



class ContrastiveCosineLoss(nn.Module):
    def __init__(self, margin: float=0.0, eps: float=1e-8, beta: float=0.8):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin
        self.eps = eps
        self.beta = beta

    def forward(self, x1: torch.tensor, x2: torch.tensor, y: torch.tensor, size_average: bool=True):
        cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        loss = (y * self.beta) * torch.abs(- (y * 1) + cos_sim(x1, x2))

        if size_average:
            return loss.mean()
        else:
            return loss.sum()