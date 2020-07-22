
import torchvision
import torch
import pickle
import numpy as np


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10):
        """ constructor

        :param list dataset: dataset list
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.dataset = dataset
        self.class_amount = class_amount

    def _preprocess_targets(self, int_representation: int, one_hot: bool=True) -> list:
        """ create one-hot encoding of the target

        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        int_representation -= 1

        if one_hot:
            one_hot_target = np.zeros((self.class_amount))
            one_hot_target[int_representation] = 1

            return one_hot_target
        else:
            return [int_representation]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][1], self.dataset[idx][0]

        # data is one-hot encoded, transform to index-representation, ie: "joe" -> [10, 15, 5], indices go from 1 ("a") to 28 ("-")
        int_name = []
        for char_oh in sample:
            int_name.append(char_oh.index(1) + 1)

        target = self._preprocess_targets(target, one_hot=False)
        
        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_sample = int_name

        return torch.Tensor(int_name), torch.Tensor(target).type(torch.LongTensor), non_padded_sample

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)

