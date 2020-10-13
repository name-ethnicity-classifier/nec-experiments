
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
from utils import device, onehot_to_string


class Model(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional

        self.embedding_size = embedding_size
        input_size = 29     # "a", "b", "c", ... " ", "-", <0 for padding>
        
        self.embed = nn.Embedding(input_size, self.embedding_size)

        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=self.bidirectional)

        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        self.linear1 = nn.Linear(self.hidden_size, 128)
        self.linear2 = nn.Linear(128, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, pad_size: int, batch_size: int, return_lstm_embeddings: bool=False):
        # init hidden state
        if self.bidirectional:
            self.layers *= 2

        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            hidden = (torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device))
        else:
            hidden = torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device)
        
        out = self.embed(x.type(torch.LongTensor).to(device=device))
        out = out.reshape(batch_size, pad_size, self.embedding_size)

        # pass input through rnn, take final output
        out, hidden = self.rnn(out, hidden)
        #print(out.shape)

        x = out[:, -1]
        lstm_embeddings = x

        # dense layers
        x = self.dropout1(x)

        x = torch.tanh(self.linear1(x))
        x = self.dropout2(x)

        x = self.logSoftmax(self.linear2(x))

        if return_lstm_embeddings:
            return lstm_embeddings, x
        else:
            return x