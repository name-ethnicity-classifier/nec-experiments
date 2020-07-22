
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import device, onehot_to_string


class Model(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, embedding_size: int=64):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

        self.embedding_size = embedding_size
        input_size = 29     # "a", "b", "c", ... " ", "-", <0 for padding>
        
        self.embed = nn.Embedding(input_size, self.embedding_size)

        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, dropout=self.dropout_chance, batch_first=True)

        self.dropout1 = nn.Dropout2d(p=(self.dropout_chance / 2))
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        self.linear1 = nn.Linear(self.hidden_size, 32)
        self.linear2 = nn.Linear(32, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, pad_size, batch_size):
        # init hidden state
        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            hidden = (torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device))
        else:
            hidden = torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device)
        
        out = self.embed(x.type(torch.cuda.LongTensor))

        out = out.reshape(batch_size, pad_size, self.embedding_size)

        # pass input through rnn, take final output
        out, hidden = self.rnn(out, hidden)
        x = out[:, -1]

        # dense layers
        x = self.dropout1(x)

        x = torch.tanh(self.linear1(x))
        x = self.dropout2(x)

        x = self.logSoftmax(self.linear2(x))

        return x