
import torch
import pickle
import numpy as np
import string
import json
import re


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10, augmentation: float=0.0):
        """ constructor

        :param list dataset: dataset list
        :param int class_amount: amount of classes(/nationalities) in the dataset
        """

        self.dataset = dataset
        # print(len(self.dataset), len(self.dataset) / 20)
        self.class_amount = class_amount

        self.augmentation = augmentation
        self.seperat_dataset = self.dataset.copy()
        np.random.shuffle(self.seperat_dataset)


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

    def _name_switch(self, org_name: list, class_: int, chance: float=0.3) -> list:
        """ switches first and last name part of the name with a random name of the same nationality """

        augmentation_choice = np.random.uniform(0.0, 1.0)
        if augmentation_choice <= chance:

            same_nat_name = []
            for idx, sample in enumerate(self.seperat_dataset):
                if class_ == sample[0]:
                    same_nat_name = [e+1 for e in sample[1]]
                
                # some names have only one part for some reason, so don't break the same-nationality search when such appear
                if 27 in same_nat_name:
                    self.seperat_dataset.pop(idx)
                    break
            
            org_prename, org_surname = self._split_name(org_name)
            same_nat_prename, same_nat_surname = self._split_name(same_nat_name)

            flip_case = np.random.choice([0, 1])

            if flip_case == 0:
                return org_prename + [27] + same_nat_surname

            elif flip_case == 1:
                return same_nat_prename + [27] + org_surname

        else:
            return org_name

    def _split_name(self, int_name: list) -> list:
        try:
            str_index_name = "".join([str(e) + " " for e in int_name])
            str_index_name_split = str_index_name.split("27", 1)

            pre_int_name, sur_int_name = str_index_name_split[0], str_index_name_split[1]
            pre_int_name = [int(e) for e in pre_int_name.split() if e.isdigit()]
            sur_int_name = [int(e) for e in sur_int_name.split() if e.isdigit()]

            return pre_int_name, sur_int_name
            
        except:
            # the case, when the name is only one word (no first-/ sur-name)
            return int_name, int_name

    def _create_n_gram(self, int_name: list, n: int=1) -> list:
        """ (please don't look at this function, it's not used and it's O(n^{inf}))
            create n-gram sample from index representation

        :param list int_name: integer/index representation of the name
        :return list: n-gram integer/index representation of the name
        """

        if n == 2:
            with open("../../datasets/ngrams/bi_gram_table.json", "r") as b:
                n_gram_table = json.load(b)
            # self.embedder = Word2Vec.load("datasets/char2vec/gensim_bigram_model.model")

        elif n == 3:
            with open("../../datasets/ngrams/tri_gram_table.json", "r") as b:
                n_gram_table = json.load(b)

        str_name = ""
        for e in int_name:
            str_name += " " + str(e)
        
        sub_names = re.split(" 27 | 28 |27 | 27|28| 28", str_name)

        for s in range(len(sub_names)):
            sub_name = [l for l in sub_names[s].split(" ") if l != ""]
            sub_names[s] = [str(l) for l in sub_name]
            
        n_gram_name = []
        for i, sub_name in enumerate(sub_names):
            # n_gram_name += [(str(l[0]) + "$" + str(l[1])) for l in list(ngrams(sub_name, n))]
            n_gram_name += ["".join([("$" + str(l[i])) for i in range(len(l))])[1:] for l in list(ngrams(sub_name, n))]

            if i != len(sub_names) - 1:
                n_gram_name += ["27"]

        n_gram_name = [n_gram_table[l] for l in n_gram_name]

        return n_gram_name

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ get sample (batch) from dataset

        :param int idx: index of dataset (iterator of training-loop)
        :return tensor: preprocessed sample and target
        """

        sample, target = self.dataset[idx][1], self.dataset[idx][0]

        int_name = [e+1 for e in sample]

        if self.augmentation > 0.0:
            int_name = self._name_switch(int_name, target, chance=self.augmentation)

        target = self._preprocess_targets(target, one_hot=False)
        
        # non_padded_batch is the original batch, which is not getting padded so it can be converted back to string
        non_padded_sample = [e+1 for e in sample]

        return torch.Tensor(int_name), torch.Tensor(target).type(torch.LongTensor), non_padded_sample    # torch.Tensor(int_name_n2), torch.Tensor(int_name_n3)

    def __len__(self):
        """ returns length of dataset """
        
        return len(self.dataset)


