
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import math
from gensim.models import Word2Vec

from utils import device, onehot_to_string




class CnnLSTM(nn.Module):
    def __init__(self, class_amount: int=0, embedding_size: int=64, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5, kernel_size: int=3, channels: list=[32, 64, 128]):
        super(CnnLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

        self.kernel_size = kernel_size
        self.channels = channels

        self.embedder = nn.Embedding(29, self.embedding_size)
        

        self.conv1 = nn.Sequential(nn.Conv1d(200, self.channels[0], kernel_size=self.kernel_size),
                                   #nn.BatchNorm1d(self.channels[0]),
                                   nn.ReLU())

        """self.conv2 = nn.Sequential(nn.Conv1d(self.channels[0], self.channels[1], kernel_size=self.kernel_size),
                                   nn.BatchNorm1d(self.channels[1]),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(self.channels[1], self.channels[2], kernel_size=self.kernel_size),
                                   nn.ReLU())"""
        
        # self.attention = nn.MultiheadAttention(self.channels[-1], num_heads=1, dropout=0.15)

        self.lstm = nn.LSTM(input_size=self.channels[-1], hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True)
        
        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, class_amount)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedder(x.type(torch.LongTensor).to(device=device))
        x = x.squeeze().transpose(1, 2)

        x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        x = x.transpose(1, 2)

        # x, _ = self.attention(x, x, x)

        hidden = (torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device), torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device))
        x, _ = self.lstm(x)
        x = x[:, -1]

        x = self.dropout(x)

        x = self.linear1(x)
        x = self.logSoftmax(x)

        return x


""" tests: """

"""a = torch.rand(1, 21)
b = torch.rand(1, 13)

model = Model(class_amount=10, hidden_size=28, dropout_chance=0.25, embedding_size=200).train()

a = model(a)
b = model(b)"""

