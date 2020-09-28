
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from termcolor import colored


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LstmEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10):
        """ constructor

        :param list dataset: dataset list
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.dataset = dataset
        self.class_amount = class_amount

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][0], self.dataset[idx][1]
        return torch.Tensor(sample), torch.Tensor(target)

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)


def create_dataloader(dataset_path: str="", batch_size: int=32, class_amount: int=10):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    dataset = np.load(dataset_path, allow_pickle=True)
    dataset = LstmEmbeddingDataset(dataset=dataset, class_amount=class_amount)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )

    return dataloader
