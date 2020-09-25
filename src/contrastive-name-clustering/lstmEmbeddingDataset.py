
import torchvision
import torch
import pickle
import numpy as np


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

