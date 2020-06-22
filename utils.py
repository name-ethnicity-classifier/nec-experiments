""" file for small helper functions """

import string
from functools import partial
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pickle
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time

from nameEthnicityDataset import NameEthnicityDataset

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def custom_collate(batch):
    """ adds custom dataloader feature: batch padding for the sample-batch (the batch containing the one-hot-enc. names)

    :param batch: three batches -> non-padded sample-batch, target-batch, non-padded sample-batch (again)
    :return torch.Tensor: padded sample-batch, target-batch, non-padded sample-batch
    """

    batch_size = len(batch)

    sample_batch, target_batch, non_padded_batch = [], [], []
    for sample, target, non_padded_sample in batch:

        if len(list(sample.size())) == 2:
            sample_batch.append(sample)
            target_batch.append(target)

            # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
            non_padded_batch.append(non_padded_sample)

    padded_batch = pad_sequence(sample_batch, batch_first=True)

    padded_to = list(padded_batch.size())[1]
    padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 28)        

    return padded_batch, torch.cat(target_batch, dim=0).reshape(len(sample_batch), target_batch[0].size(0)), non_padded_batch

def create_dataloader(dataset_path: str="", test_size: float=0.01, val_size: float=0.01, batch_size: int=32, class_amount: int=10):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    dataset = NameEthnicityDataset(root_dir=dataset_path, class_amount=class_amount)

    test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
        (dataset.__len__() - (test_amount + val_amount)), 
        test_amount, 
        val_amount
    ])

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
        collate_fn=custom_collate

    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
        collate_fn=custom_collate
    )

    return train_dataloader, val_dataloader, test_dataloader


def validate_accuracy(y_true, y_pred, threshold: float) -> float:
    """ calculate the accuracy of predictions
    
    :param torch.tensor y_true: targets
    :param torch.tensor y_pred: predictions
    :param float threshold: treshold for logit-rounding
    :return float: accuracy
    """

    correct_in_batch = 0
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]

        amount_classes = output.shape[0]

        target_empty = np.zeros((amount_classes))
        target_empty[target] = 1
        target = target_empty

        output = list(output).index(max(output))
        output_empty = np.zeros((amount_classes))
        output_empty[output] = 1
        output = output_empty

        # output = list(np.exp(output))
        # output = [1 if e >= threshold else 0 for e in output]

        if list(target) == list(output):
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


def show_progress(epochs: int, epoch: int, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):
    """ print training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float train_loss/train_accuracy: train-loss, train-accuracy
    :param float val_loss/val_accuracy: validation accuracy/loss
    :return None
    """

    epochs = colored(epoch, "cyan", attrs=["bold"]) + colored("/", "cyan", attrs=["bold"]) + colored(epochs, "cyan", attrs=["bold"])
    train_accuracy = colored(round(train_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    train_loss = colored(round(train_loss, 6), "cyan", attrs=["bold"])
    val_accuracy = colored(round(val_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=["bold"])
    
    print("epoch {} train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}".format(epochs, train_loss, train_accuracy, val_loss, val_accuracy), "\n")


def onehot_to_string(one_hot_name: list=[]) -> str:
    """ convert one-hot encoded name back to string

    :param list one_hot_name: one-hot enc. name
    :return str: original string-type name
    """

    alphabet = string.ascii_lowercase.strip()

    name = ""
    for one_hot_char in one_hot_name:
        idx = list(one_hot_char).index(1) # [i for i, x in enumerate(one_hot_char) if x == 1][0]

        if idx == 26:
            name += " "
        elif idx == 27:
            name += "-"
        else:
            name += alphabet[idx]

    return name


def string_to_onehot(string_name: str="") -> list:
    """ create one-hot encoded name

    :param str name: name to encode
    :return list: list of all one-hot encoded letters of name
    """

    alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]

    full_name_onehot = []
    for char in string_name:
        char_idx = alphabet.index(char)

        one_hot_char = np.zeros((28))
        one_hot_char[char_idx] = 1

        full_name_onehot.append(one_hot_char)
    
    return full_name_onehot


def init_xavier_weights(m):
    """ initializes model parameters with xavier-initialization

    :param m: model parameters
    """
    if isinstance(m, nn.RNN):
        nn.init.xavier_uniform_(m.weight_hh_l0.data)


def plot(train_acc: list, train_loss: list, val_acc: list, val_loss: list, save_to: str=""):
    """ plots training stats
    
    :param list train_acc/train_loss: training accuracy and loss
    :param list val_acc/val_loss: validation accuracy and loss
    """

    plt.style.use("ggplot")
    fig, axs = plt.subplots(2)
    xs = range(1, (len(train_acc) + 1))

    axs[0].plot(xs, train_acc, "r", label="train-acc")
    axs[0].plot(xs, val_acc, "b", label="val-acc")
    axs[0].legend()
    axs[0].set_title("train-/ val-acc")

    axs[1].plot(xs, train_loss, "r", label="train-loss")
    axs[1].plot(xs, val_loss, "b", label="val-loss")
    axs[1].legend()
    axs[1].set_title("train-/ val-loss")
    
    if save_to != "":
        plt.savefig(save_to)

    plt.show()


