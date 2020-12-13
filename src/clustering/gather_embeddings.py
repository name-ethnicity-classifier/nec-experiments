
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from nameEthnicityDataset import NameEthnicityDataset
from model import Model
from utils import create_dataloader, show_progress, onehot_to_string, init_xavier_weights, device, char_indices_to_string



with open("../datasets/final_nationality_to_number_dict.json", "r") as f: classes = json.load(f) 
total_classes = len(classes)

hidden_size = 256
layers = 2
dropout_chance = 0.0
embedding_size = 128
batch_size = 512
dataset_path = "../datasets/final_matrix_name_list.pickle"
model_file = "../models/model7.pt"

embeddings_dataset_file = "dataset/lstm_embeddings_test.npy"


def get_embeddings():
    model = Model(class_amount=total_classes, hidden_size=hidden_size, layers=layers, dropout_chance=dropout_chance, embedding_size=embedding_size).to(device=device)
    model.load_state_dict(torch.load(model_file))

    train_set, validation_set, test_set = create_dataloader(dataset_path=dataset_path, test_size=0.025, val_size=0.025, batch_size=batch_size, class_amount=total_classes, shuffle=False)
    
    total_lstm_embeddings = []
    for names, targets, _ in tqdm(test_set, desc="", ncols=150):
        names, targets = names.to(device=device), targets.to(device=device)

        lstm_embeddings, _ = model.eval()(names, len(names[0]), len(names), return_lstm_embeddings=True)

        for idx in range(lstm_embeddings.shape[0]):
            lstm_embedding = lstm_embeddings[idx].cpu().detach().numpy()
            target = targets[idx].cpu().detach().numpy()

            total_lstm_embeddings.append([lstm_embedding, target])

    np.save(embeddings_dataset_file, total_lstm_embeddings, allow_pickle=True)


get_embeddings()