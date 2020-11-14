
import torchvision
import torch
import pickle
import numpy as np
import string
from nltk import ngrams
import json
import re

class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10, augmentation: bool=False, bi_gram: bool=False):
        """ constructor

        :param list dataset: dataset list
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.dataset = dataset
        self.class_amount = class_amount

        self.augmentation = augmentation
        self.bi_gram = bi_gram

        if self.bi_gram:
            with open("datasets/bi_gram_table.json", "r") as b:
                self.bi_gram_table = json.load(b)

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

    def _augmentate(self, int_name: list) -> list:
        """ augmentate name by either flipping sure- and prename or just by taking the surname

        :paran list int_name: integer/index representation of the name
        :return list: augmentated integer/index representation of the name
        """

        augmentation_choice = np.random.choice([1, 2, 3, 4, 5])

        int_rep_prename, int_rep_surname = [], []
        is_prename = True
        for int_rep_char in int_name:

            # check if the ineteger representates the space-symbol
            if int_rep_char == 27:
                is_prename = False
                continue

            if is_prename:
                int_rep_prename.append(int_rep_char)
            else:
                int_rep_surname.append(int_rep_char)

        # only surname
        if augmentation_choice == 1:
            return int_rep_surname
                
        # flip surename with prename
        elif augmentation_choice == 2:
            return int_rep_surname + [27] + int_rep_prename

        # return normale prename + surename
        else:
            return int_name

    def _create_bi_gram(self, int_name: list) -> list:
        """ create bi-gram sample from index representation

        :param list int_name: integer/index representation of the name
        :return list: bi-gram integer/index representation of the name
        """

        str_name = ""
        for e in int_name:
            str_name += " " + str(e)
        
        sub_names = re.split(" 27 | 28 |27 | 27|28| 28", str_name)

        for s in range(len(sub_names)):
            sub_name = [l for l in sub_names[s].split(" ") if l != ""]
            sub_names[s] = [str(l) for l in sub_name]
            
        bi_gram_name = []
        for i, sub_name in enumerate(sub_names):
            bi_gram_name += [(str(l[0]) + "$" + str(l[1])) for l in list(ngrams(sub_name, 2))]
            if i != len(sub_names) - 1:
                bi_gram_name += ["27"]

        bi_idx_name = [self.bi_gram_table[l] for l in bi_gram_name]

        return bi_idx_name

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][1], self.dataset[idx][0]

        # data is one-hot encoded, transform to index-representation, ie: "joe" -> [10, 15, 5], indices go from 1 ("a") to 28 ("-")
        """int_name = []
        for char_one_hot in sample:
            int_name.append(char_one_hot.index(1) + 1)"""

        int_name = [e+1 for e in sample]

        if self.augmentation:
            int_name = self._augmentate(int_name)

        # if bi-gram should be used
        if self.bi_gram:
            int_name = self._create_bi_gram(int_name)
            # print(int_name, "\n____\n")

        target = self._preprocess_targets(target, one_hot=False)
        
        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_sample = [e+1 for e in sample]

        return torch.Tensor(int_name), torch.Tensor(target).type(torch.LongTensor), non_padded_sample

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)

