
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CoordinateModel(nn.Module):
    def __init__(self, input_size: int=64, dropout_chance: float=0.5):
        super(CoordinateModel, self).__init__()
        self.input_size = input_size
        self.dropout_chance = dropout_chance

        self.linear1 = nn.Linear(self.input_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 3)

        self.dropout = nn.Dropout2d(p=self.dropout_chance)

        """self.linear1 = nn.Linear(self.input_size, 512)
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)

        self.linear2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        self.linear3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout2d(p=self.dropout_chance)

        self.linear4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout2d(p=(self.dropout_chance / 2))

        # output: x and y coordinates ( range ]-1, 1[ )
        self.linear5 = nn.Linear(64, 2) """

    def forward(self, x):
        """x = F.relu(self.linear1(x))
        x = self.dropout1(x)

        x = F.relu(self.linear2(x))
        x = self.dropout2(x)

        x = F.relu(self.linear3(x))
        x = self.dropout3(x)

        x = F.relu(self.linear4(x))
        x = self.dropout4(x)

        x = self.linear5(x)"""
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.linear4(x)

        return x