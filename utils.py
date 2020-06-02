""" file for small helper functions """

import string
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn


def create_dataloader(dataset_path: str="", test_size: float=0.01, val_size: float=0.01, batch_size: int=32):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    dataset = NameEthnicityDatset(dataset_path)

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
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size / 2),
        num_workers=1,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        num_workers=1,
        shuffle=True,
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
        output = [1 if e >= threshold else 0 for e in output]

        if list(target) == output:
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


def show_progress(epochs: int, epoch: int, loss: float, val_accuracy: float, val_loss: float):
    """ print training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float loss: train-loss
    :param float val_accuracy/val_loss: validation accuracy/loss
    :return None
    """
    epochs = colored(epoch, "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    loss = colored(round(loss, 6), "cyan", attrs=['bold'])
    accuracy = colored(round(val_accuracy, 4), "cyan", attrs=['bold']) + colored("%", "cyan", attrs=['bold'])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=['bold'])
    
    print("epoch {} - loss: {} - val_acc: {} - val_loss: {}".format(epochs, loss, accuracy, val_loss), "\n")


def onehot_to_char(one_hot_name: list=[]) -> str:
    alphabet = string.ascii_lowercase.strip()
    """ convert one-hot encoded name back to string

    :param list one_hot_name: one-hot enc. name
    :return str: original string-type name
    """

    name = ""
    for one_hot_char in one_hot_name:
        idx = [i for i, x in enumerate(one_hot_char) if x == 1][0]

        if idx == 26:
            name += " "
        elif idx == 27:
            name += "-"
        else:
            name += alphabet[idx]

    return name


""" onehot_to_char test

j = np.zeros((28))
j[9] = 1

o = np.zeros((28))
o[14] = 1

e = np.zeros((28))
e[4] = 1

leer = np.zeros((28))
leer[26] = 1

a = np.zeros((28))
a[0] = 1

b = np.zeros((28))
b[1] = 1

c = np.zeros((28))
c[2] = 1

test_name = [j, o, e, leer, a, b, c]
print(onehot_to_char(one_hot_name=test_name))"""