
import torchvision
import torch
import json
import numpy as np


class NameEthnicityDatset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str="", class_amount: int=10):
        self.root_dir = root_dir
        self.class_amount = class_amount

        with open(self.root_dir, "r") as f:
            self.dataset = json.load(f)
    
    def _create_one_hot(self, int_representation: int) -> list:
        zero_hot_target = np.zeros((self.class_amount))
        one_hot_target = empty_target[int_representation] = 1

        return one_hot_target

    def __getitem__(self, idx: int):
        sample, target = self.dataset[idx][0], self.dataset[idx][1]
        target = self._create_one_hot(target)

        return torch.Tensor(sample), torch.Tensor(target)



# docstring
""" creates one-hot representation of class 

:param int_representation: class number n [0; class_amount-1]
:type int_representation: int
:rtype: list
:return: ie, class_amount = 5 ; int_representation = 1 -> [0, 1, 0, 0, 0]
"""