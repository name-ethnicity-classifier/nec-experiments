
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import math

from utils import device, onehot_to_string



""" LSTM """
class LSTM(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional

        self.embedding_size = embedding_size
        vocab_size = pow(26, 3) + 2   # 677 + 1     # "a", "b", "c", ... " ", "-", <0 for padding>
        
        self.embed = nn.Embedding(vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=self.bidirectional)

        directions = 1
        if self.bidirectional:
            self.layers *= 2
            directions = 2

        self.dropout0 = nn.Dropout2d(p=0.15)
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        self.linear1 = nn.Linear(self.hidden_size*directions, class_amount)
        # self.linear2 = nn.Linear(128, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1) 


    def forward(self, x, pad_size: int, batch_size: int, return_lstm_embeddings: bool=False):
        # init hidden state
        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            hidden = (torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device))
        else:
            hidden = torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device)

        x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = x.reshape(batch_size, pad_size, self.embedding_size)
        # out = self.dropout0(out)

        # pass input through rnn, take final output
        out, hidden = self.rnn(x, hidden)

        x = out[:, -1]
        lstm_embeddings = x

        # dense layers
        x = self.dropout1(x)

        # x = torch.tanh(self.linear1(x))
        # x = self.dropout2(x)

        x = self.logSoftmax(self.linear1(x))

        if return_lstm_embeddings:
            return lstm_embeddings, x
        else:
            return x





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






""" ATTENTION -> BI-LSTM """
class AttentionModel(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(AttentionModel, self).__init__()

        self.class_amount = class_amount
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0

        vocab_size = pow(26, 3) + 2
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.pe = PositionalEncoder(self.embedding_size)

        self.attention = nn.MultiheadAttention(self.embedding_size, num_heads=1, dropout=0.35)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(self.hidden_size*2, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1) 
        
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        
    def forward(self, x, pad_size: int, batch_size: int, return_lstm_embeddings: bool=False):
        mask = (x == 0).cuda()

        # embedding layer
        x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = self.pe(x)
        #x = x.reshape(batch_size, pad_size, self.embedding_size)

        # multi-head-attention layer
        x, _ = self.attention(x, x, x)
        x = x.reshape(batch_size, pad_size, self.embedding_size)

        # bi-directional lstm layer
        hidden = (torch.zeros((self.layers * 2), batch_size, self.hidden_size).to(device=device), torch.zeros((self.layers * 2), batch_size, self.hidden_size).to(device=device))
        x, hidden = self.bi_lstm(x, hidden)
        x = x.reshape(batch_size, pad_size, (self.hidden_size * 2))
        x = x[:, -1]

        x = self.logSoftmax(self.linear1(x))

        return x






""" BI-LSTM -> ATTENTION """
class BiLstmAttentionModel(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(BiLstmAttentionModel, self).__init__()

        self.class_amount = class_amount
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0

        vocab_size = pow(26, 3) + 2
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.pe = PositionalEncoder(self.embedding_size)

        self.bi_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention((self.hidden_size * 2), num_heads=1, dropout=0.35)

        self.rnn = nn.RNN(input_size=(self.hidden_size * 2), hidden_size=self.class_amount, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=False)

        self.linear1 = nn.Linear(self.hidden_size*2, class_amount)

        self.logSoftmax = nn.LogSoftmax(dim=1) 
        
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)
        self.dropout2 = nn.Dropout2d(p=self.dropout_chance)

        
    def forward(self, x, pad_size: int, batch_size: int, return_lstm_embeddings: bool=False):
        mask = (x == 0).cuda()
        hidden = (torch.zeros((self.layers * 2), batch_size, self.hidden_size).to(device=device), torch.zeros((self.layers * 2), batch_size, self.hidden_size).to(device=device))

        # embedding layer
        x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = x.reshape(batch_size, pad_size, self.embedding_size)

        # bi-directional lstm layer
        x, hidden = self.bi_lstm(x, hidden)

        # split forward and backward lstm outputs
        #x1, x2 = x[:, :, :int(x.shape[2]/2)], x[:, :, int(x.shape[2]/2):]
        # merging by adding
        #x = x1 + x2

        # save last output
        x_final = x[:, -1]

        # apply activation
        x = F.gelu(x)
        x = x.reshape(pad_size, batch_size, (self.hidden_size * 2))

        # multi-head-attention layer
        x, _ = self.attention(x, x, x)
        x = x.reshape(batch_size, pad_size, (self.hidden_size * 2))

        # post attention gru layer
        #hidden = torch.zeros(self.layers, batch_size, self.class_amount).to(device=device)
        #x, hidden = self.rnn(x, hidden)
        #x = x[:, -1]
        #x = self.dropout1(x)
        #x = x.sum(1) + x_final
        #x = x.mean(1) + x_final
        x = x.mean(1) + x_final

        x = self.logSoftmax(self.linear1(x))

        return x




""" TRANSFORMER -> LSTM """
class TransformerLSTMModel(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(TransformerLSTMModel, self).__init__()

        self.class_amount = class_amount
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional

        vocab_size = pow(26, 3) + 2
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.pe = PositionalEncoder(self.embedding_size)

        if self.bidirectional:
            self.layers *= 2
            self.directions = 2
        else:
            self.directions = 1

        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_size, nhead=1, dropout=0.35)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, 
                            dropout=self.lstm_dropout_chance, batch_first=True, bidirectional=self.bidirectional)

        self.linear1 = nn.Linear(self.hidden_size*self.directions, class_amount)
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)

        self.logSoftmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, x, pad_size: int, batch_size: int):
        mask = (x == 0).cuda()

        # embedding layer
        # x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = self.pe(x)
        #x = x.reshape(batch_size, pad_size, self.embedding_size)

        # multi-head-attention layer
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, pad_size, self.embedding_size)
        
        # bi-directional lstm layer
        if isinstance(self.rnn, torch.nn.modules.rnn.LSTM):
            hidden = (torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device), torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device))
        else:
            hidden = torch.zeros(self.layers, batch_size, self.hidden_size).to(device=device)
        
        x, hidden = self.rnn(x, hidden)
        x = x.reshape(batch_size, pad_size, (self.hidden_size*self.directions))
        x = x[:, -1]

        x = self.dropout1(x)

        x = self.logSoftmax(self.linear1(x))

        return x






""" TRANSFORMER -> MEAN """
class TransformerModel(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(TransformerModel, self).__init__()

        self.class_amount = class_amount
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional

        vocab_size = pow(26, 3) + 2
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.pe = PositionalEncoder(self.embedding_size)

        self.encoder_layer = nn.TransformerEncoderLayer(self.embedding_size, nhead=1, dropout=0.4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=5)

        self.linear1 = nn.Linear(self.embedding_size, class_amount)
        self.dropout1 = nn.Dropout2d(p=self.dropout_chance)

        self.logSoftmax = nn.LogSoftmax(dim=1) 
        
    def forward(self, x, pad_size: int, batch_size: int):
        mask = (x == 0).cuda()

        # embedding layer
        x = self.embed(x.type(torch.LongTensor).to(device=device))
        x = self.pe(x)
        #x = x.reshape(batch_size, pad_size, self.embedding_size)

        # multi-head-attention layer
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, pad_size, self.embedding_size)
        
        x = x.mean(1)
        # x = maxpool
        # x = self.dropout1(x)

        x = self.logSoftmax(self.linear1(x))

        return x




""" N-GRAM LSTM """


class SingleLstm(nn.Module):
    def __init__(self, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(SingleLstm, self).__init__()

        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.layers, dropout=self.lstm_dropout_chance, batch_first=True)

    def forward(self, x):
        hidden = (torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device), torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device))
        x, hidden = self.rnn(x, hidden)

        return x[:, -1]

class TripleNGramLSTM(nn.Module):
    def __init__(self, class_amount: int=0, hidden_size: int=10, layers: int=2, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64):
        super(TripleNGramLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.lstm_dropout_chance = dropout_chance if layers > 1 else 0.0
        self.bidirectional = bidirectional

        self.embedding_size = embedding_size

        vocab_size_n1 = pow(26, 1) + 2
        self.embed_n1 = nn.Embedding(vocab_size_n1, self.embedding_size)

        vocab_size_n2 = pow(26, 2) + 2
        self.embed_n2 = nn.Embedding(vocab_size_n2, self.embedding_size)

        vocab_size_n3 = pow(26, 3) + 2
        self.embed_n3 = nn.Embedding(vocab_size_n3, self.embedding_size)

        self.lstm_n1 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.embedding_size)
        self.lstm_n2 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.embedding_size)
        self.lstm_n3 = SingleLstm(self.hidden_size, self.layers, self.lstm_dropout_chance, self.embedding_size)

        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, class_amount)
        self.logSoftmax = nn.LogSoftmax(dim=1) 


    def forward(self, x_n1, x_n2, x_n3):
        x_n1 = self.embed_n1(x_n1.type(torch.LongTensor).to(device=device)).reshape(x_n1.size(1), x_n1.size(0), self.embedding_size)
        x_n2 = self.embed_n2(x_n2.type(torch.LongTensor).to(device=device)).reshape(x_n2.size(1), x_n2.size(0), self.embedding_size)
        x_n3 = self.embed_n3(x_n3.type(torch.LongTensor).to(device=device)).reshape(x_n3.size(1), x_n3.size(0), self.embedding_size)

        x_n1 = self.lstm_n1(x_n1)
        x_n2 = self.lstm_n1(x_n2)
        x_n3 = self.lstm_n1(x_n3)

        x = torch.cat((x_n1, x_n2, x_n3), 2)

        x = self.linear1(x)
        x = self.logSoftmax()

        return x