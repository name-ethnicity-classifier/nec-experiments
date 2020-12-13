
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import math
from gensim.models import Word2Vec

from utils import device, onehot_to_string



""" N-GRAM LSTM """


class SingleLstm(nn.Module):
    def __init__(self, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(SingleLstm, self).__init__()

        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.embedding_size = embedding_size

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, \
                            dropout=self.lstm_dropout_chance, bidirectional=self.bidirectional, batch_first=True)

        if self.bidirectional: self.layers *= 2

    def forward(self, x):
        hidden = (torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device), torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device))
        x, hidden = self.lstm(x, hidden)

        return x


class TripleNGramLSTM(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(TripleNGramLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.embedding_size = embedding_size

        self.directions = 2 if self.bidirectional else 1

        self.embedder = nn.Embedding(pow(26, 3) + pow(26, 2) + 26 + 9, self.embedding_size)

        self.lstm_n1 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.bidirectional, self.embedding_size)
        self.lstm_n2 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.bidirectional, self.embedding_size)
        self.lstm_n3 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.bidirectional, self.embedding_size)

        self.attention1 = nn.MultiheadAttention(self.hidden_size * self.directions, num_heads=1, dropout=0.35)
        self.attention2 = nn.MultiheadAttention(self.hidden_size * self.directions, num_heads=1, dropout=0.35)
        self.attention3 = nn.MultiheadAttention(self.hidden_size * self.directions, num_heads=1, dropout=0.35)

        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size * 3 * self.directions, 200 * self.directions)
        self.linear2 = nn.Linear(200 * self.directions, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, x_n1, x_n2, x_n3):
        x_n1 = self.embedder(x_n1.type(torch.LongTensor).to(device=device)).reshape(x_n1.size(0), x_n1.size(1), self.embedding_size)
        x_n2 = self.embedder(x_n2.type(torch.LongTensor).to(device=device)).reshape(x_n2.size(0), x_n2.size(1), self.embedding_size)
        x_n3 = self.embedder(x_n3.type(torch.LongTensor).to(device=device)).reshape(x_n3.size(0), x_n3.size(1), self.embedding_size)

        x_n1 = self.lstm_n1(x_n1)#.reshape(x_n1.size(0), x_n1.size(1), self.hidden_size * self.directions)
        x_n2 = self.lstm_n2(x_n2)#.reshape(x_n1.size(0), self.hidden_size * self.directions)
        x_n3 = self.lstm_n3(x_n3)#.reshape(x_n1.size(0), self.hidden_size * self.directions)
        
        x_n1_last = x_n1[:, -1]
        x_n2_last = x_n2[:, -1]
        x_n3_last = x_n3[:, -1]

        x_n1, _ = self.attention1(x_n1, x_n1, x_n1)
        x_n2, _ = self.attention2(x_n2, x_n2, x_n2)
        x_n3, _ = self.attention3(x_n3, x_n3, x_n3)

        x_n1 = x_n1.mean(1) + x_n1_last
        x_n2 = x_n2.mean(1) + x_n2_last
        x_n3 = x_n3.mean(1) + x_n3_last

        x = torch.cat((x_n1, x_n2, x_n3), -1)

        # x = torch.cat((x_n1, x_n2, x_n3), -2)
        # x, _ = self.attention(x, x, x)
        # x = x.mean(1)

        x = F.relu(self.linear1(x))

        x = self.logSoftmax(self.linear2(x))

        return x