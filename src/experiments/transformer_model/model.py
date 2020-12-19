
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import math

from utils import device, onehot_to_string



""" POSITIONAL-ENCODER MODULE """
class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        seq_len = x.size(1)

        pe = torch.zeros(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))

        x = x * math.sqrt(self.d_model)
        x = x.squeeze()

        x = x + pe.requires_grad_(False).cuda()

        return x


class TransformerModel(nn.Module):
    def __init__(self, class_amount: int=0, num_heads: int=4, num_layers: int=6, dropout_chance: float=0.5, embedding_size: int=64):
        super(TransformerModel, self).__init__()

        self.class_amount = class_amount
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_chance = dropout_chance

        vocab_size = pow(26, 3) + 2
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.pe = PositionalEncoder(self.embedding_size)

        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_size, nhead=self.num_heads, dropout=self.dropout_chance)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.linear1 = nn.Linear(self.embedding_size, self.embedding_size // 2)
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance / 2)

        self.linear2 = nn.Linear(self.embedding_size // 2, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, x):
        mask = (x == 0).cuda()

        # embedding layer
        x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = self.pe(x)

        # multi-head-attention layer
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), x.size(1), self.embedding_size)
        
        x = x.mean(1)

        x = F.gelu(self.linear1(x))
        x = self.dropout1(x)
        x = self.logSoftmax(self.linear2(x))

        return x

