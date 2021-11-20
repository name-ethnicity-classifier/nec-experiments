

import torch
import pickle5 as pickle
import numpy as np
import json

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)

    
def write_json(file_path: str, content: dict) -> None:
    with open(file_path, "w") as f:
            json.dump(content, f, indent=4)


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10):
        self.dataset = dataset
        self.class_amount = class_amount

    def __getitem__(self, idx: int) -> torch.Tensor:
        target, sample, gender, year_of_birth = self.dataset[idx][0], self.dataset[idx][1], self.dataset[idx][2], self.dataset[idx][3]
        int_name = [(e + 1) for e in sample]

        return torch.Tensor(int_name), torch.Tensor([target]).type(torch.LongTensor), gender, year_of_birth

    def __len__(self):
        return len(self.dataset)


def create_dataloader(dataset_path: str="", class_amount: int=10):
    """ create three dataloader (train, test, validation)

    :param str dataset_path: path to dataset
    :param int class_amount: amount of classes
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    dataset = NameEthnicityDataset(dataset=dataset, class_amount=class_amount)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    return dataloader