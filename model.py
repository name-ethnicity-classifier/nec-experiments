
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_size = 512
        self.layers = 3
        self.dropout_chance = 0.4

        self.rnn = nn.RNN(input_size=28, hidden_size=self.hidden_size, num_layers=self.layers, dropout=self.dropout_chance, batch_first=True)
        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, 200)
        self.linear2 = nn.Linear(200, 42)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, pad_size, batch_size):
        hidden = torch.zeros(self.layers, batch_size, self.hidden_size).cuda()

        out, hidden = self.rnn(x, hidden)

        # take final hidden state output
        x = out[:, -1]

        x = self.dropout(x)

        # dense layers
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        #x = F.softmax(x, dim=1)
        x = self.logSoftmax(x)

        return x


"""test = torch.rand((1, 16, 28)).cuda()

model = Model().cuda()
output = model.train()(test, 16)
print(output.size())"""