
import torchvision
import torch
import pickle
import numpy as np


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str="", class_amount: int=10):
        """ constructor

        :param str root_dir: dataset path
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.root_dir = root_dir
        self.class_amount = class_amount

        with open(self.root_dir, "rb") as f:
            self.dataset = pickle.load(f)#[:1000]
    
    def _create_one_hot(self, int_representation: int) -> list:
        """ create one-hot encoding of the target

        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        int_representation -= 1
        #one_hot_target = np.zeros((self.class_amount))
        #one_hot_target[int_representation] = 1

        #return one_hot_target
        return [int_representation]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][1], self.dataset[idx][0]
        target = self._create_one_hot(target)
        
        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_sample = sample

        return torch.Tensor(sample), torch.Tensor(target).type(torch.LongTensor), non_padded_sample

    def __len__(self):
        return len(self.dataset)
