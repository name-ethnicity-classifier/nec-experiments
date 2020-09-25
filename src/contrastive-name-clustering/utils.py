
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from termcolor import colored

from lstmEmbeddingDataset import LstmEmbeddingDataset

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataloader(dataset_path: str="", test_size: float=0.01, val_size: float=0.01, batch_size: int=32, class_amount: int=10):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    dataset = np.load(dataset_path, allow_pickle=True)

    test_size = int(np.round(len(dataset)*test_size))
    val_size = int(np.round(len(dataset)*val_size))

    train_set, test_set, validation_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    train_set = LstmEmbeddingDataset(dataset=train_set, class_amount=class_amount)
    test_set = LstmEmbeddingDataset(dataset=test_set, class_amount=class_amount)
    val_set = LstmEmbeddingDataset(dataset=validation_set, class_amount=class_amount)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(batch_size),
        num_workers=1,
        shuffle=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


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
