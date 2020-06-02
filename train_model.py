""" file to train and evaluate the model """

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.utils.data
import torch.nn as nn

from nameEthnicityDataset import NameEthnicityDatset
from model import Model
from utils import create_dataloader, validate_accuracy, show_progress, onehot_to_char


class Run:
    def __init__(self, model_file: str="", dataset_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=32, threshold: float=0.5):
        self.model_file = model_file
        self.dataset_path = dataset_path
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=0.01, val_size=0.01, batch_size=self.batch_size)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.threshold = threshold

    def _prepare_batch(self, batch):
        """ TODO add batch-padding """

        return batch

    def _validate(self, model, dataset):
        validation_dataset = dataset

        total_targets, total_predictions = [], []

        for names, targets in tqdm(validation_dataset, desc="validating", ncols=150):
            names = names.float().cuda()
            targets = targets.float().cuda()

            predictions = model.eval()(names)

            for i in range(predictions.size()[0]):
                total_targets.append(targets[i].cpu().detach().numpy())
                total_predictions.append(predictions[i].cpu().detach().numpy())

        # calculate accuracy
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=self.threshold)

        # calculate loss
        criterion = nn.MSELoss()
        loss = criterion(predictions, targets).item()

        return loss, accuracy

    def train(self, continue_: bool=False):
        model = Model.cuda()

        if continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = [], [], [], []
        for epoch in range(1, (self.epochs + 1)):

            epoch_train_loss = []
            for names, targets in tqdm(self.train_set, desc="epoch", ncols=150):
                optimizer.zero_grad()

                names, targets = names.cuda(), targets.cuda()
                names = self._prepare_batch(names)

                predictions = model.train()(names)

                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_train_loss.append(loss.item())
        
            epoch_train_loss, epoch_train_accuracy = np.mean(epoch_loss), self._validate(model, self.train_set)
            epoch_val_loss, epoch_val_accuracy = self._validate(model, self.validation_set)

            train_loss_history.append(epoch_train_loss); train_accuracy_history.append(epoch_train_accuracy)
            val_loss_history.append(epoch_val_loss); val_accuracy_history.append(epoch_val_accuracy)

            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)

        # TODO add history plot

    def test(self):
        model = Model.cuda()
        model.load_state_dict(torch.load(self.model_file))

        accuracy, _ = self._validate(model, self.test_set)

        print(accuracy)
