
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import device


class Model(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

        self.rnn = nn.GRU(input_size=28, hidden_size=self.hidden_size, num_layers=self.layers, dropout=self.dropout_chance, batch_first=True)
        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, 200)
        self.linear2 = nn.Linear(200, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, pad_size, batch_size):
        # init hidden state

        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            hidden = (torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device))
        else:
            hidden = torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device)
        
        # pass input through rnn, take final output
        out, hidden = self.rnn(x, hidden)
        x = out[:, -1]

        # dense layers
        x = self.dropout(x)

        x = torch.tanh(self.linear1(x))
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.logSoftmax(x)

        return x

"""
test = torch.rand((1, 16, 28)).to(device=device)

model = Model(class_amount=33, hidden_size=1, layers=2, dropout_chance=0.5).to(device=device)
output = model.train()(test, 16, 1)
print(output.size())
"""
