""" file to train and evaluate the model """

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from nameEthnicityDataset import NameEthnicityDataset
from model import Model
from utils import create_dataloader, show_progress, onehot_to_string, init_xavier_weights, device, char_indices_to_string
from test_metrics import validate_accuracy, create_confusion_matrix

import xman

torch.manual_seed(0)


with open("datasets/final_nationality_to_number_dict.json", "r") as f: classes = json.load(f) 
total_classes = len(classes)


class Run:
    def __init__(self, model_file: str="", dataset_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=32, \
                    threshold: float=0.5, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5, embedding_size: int=64, continue_: bool=False):
        self.model_file = model_file
        self.dataset_path = dataset_path
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=0.025, val_size=0.025, batch_size=batch_size, class_amount=total_classes, shuffle=(not continue_))

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.threshold = threshold

        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance
        self.embedding_size = embedding_size

        self.continue_ = continue_

        self.xmanager = xman.ExperimentManager(experiment_name="experiment6", new=False, continue_=self.continue_)
        self.xmanager.init(optimizer="Adam", 
                            loss_function="NLLLoss",
                            epochs=self.epochs, 
                            learning_rate=self.lr, 
                            batch_size=self.batch_size,
                            custom_parameters={
                                "lstm": 1,
                                "hidden-size": self.hidden_size,
                                "layers": self.layers,
                                "dropout-chance": self.dropout_chance,
                                "embedding-size": self.embedding_size,
                                "dense-layer-1": "tanh",
                                "dense-layer-2": "logsoftmax",
                            })

    def _validate(self, model, dataset, confusion_matrix: bool=False):
        validation_dataset = dataset

        criterion = nn.NLLLoss()
        losses = []
        total_targets, total_predictions = [], []

        for names, targets, _ in tqdm(validation_dataset, desc="validating", ncols=150):
            names, targets = names.to(device=device), targets.to(device=device)

            predictions = model.eval()(names, len(names[0]), len(names))
            loss = criterion(predictions, targets.squeeze())
            losses.append(loss.item())

            for i in range(predictions.size()[0]):
                total_targets.append(targets[i].cpu().detach().numpy())
                total_predictions.append(predictions[i].cpu().detach().numpy())

        # calculate accuracy
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=self.threshold)

        if confusion_matrix:
            create_confusion_matrix(total_targets, total_predictions, classes=classes)

        # calculate loss
        loss = np.mean(losses)

        return loss, accuracy

    def train(self):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance, embedding_size=self.embedding_size).to(device=device)
        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = [], [], [], []
        for epoch in range(1, (self.epochs + 1)):

            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []
            for names, targets, _ in tqdm(self.train_set, desc="epoch", ncols=150):
                optimizer.zero_grad()

                names, targets = names.to(device=device), targets.to(device=device)

                predictions = model.train()(names, len(names[0]), len(names))

                loss = criterion(predictions, targets.squeeze())
                loss.backward()
                optimizer.step()

                # log train loss
                epoch_train_loss.append(loss.item())
                
                # log targets and prediction of every iteration to compute the accuracy later
                validated_predictions = model.eval()(names, len(names[0]), len(names))
                for i in range(validated_predictions.size()[0]): total_train_targets.append(targets[i].cpu().detach().numpy()); \
                                                        total_train_predictions.append(validated_predictions[i].cpu().detach().numpy())

            # calculate train loss and accuracy of last epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = validate_accuracy(total_train_targets, total_train_predictions, threshold=self.threshold)

            # calculate validation loss and accuracy of last epoch
            epoch_val_loss, epoch_val_accuracy = self._validate(model, self.validation_set)

            # log training stats
            train_loss_history.append(epoch_train_loss); train_accuracy_history.append(epoch_train_accuracy)
            val_loss_history.append(epoch_val_loss); val_accuracy_history.append(epoch_val_accuracy)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)

            # save checkpoint of model
            torch.save(model.state_dict(), self.model_file)

            # log epoch results with xman
            self.xmanager.log_epoch(model, self.lr, self.batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

        # plot train-history with xman
        self.xmanager.plot_history(save=True)

    def test(self):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance, embedding_size=self.embedding_size).to(device=device)
        model.load_state_dict(torch.load(self.model_file))

        _, accuracy = self._validate(model, self.test_set, confusion_matrix=True)

        for names, targets, non_padded_names in tqdm(self.test_set, desc="epoch", ncols=150):
            names, targets = names.to(device=device), targets.to(device=device)

            predictions = model.eval()(names, len(names[0]), len(names))
            predictions, targets, names = predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), names.cpu().detach().numpy()

            for idx in range(len(names)):
                name, prediction, target, non_padded_name = names[idx], predictions[idx], targets[idx], non_padded_names[idx]

                # convert to one-hot target
                amount_classes = prediction.shape[0]
                target_empty = np.zeros((amount_classes))
                target_empty[target] = 1
                target = target_empty

                # convert log-softmax to one-hot
                prediction = list(np.exp(prediction))
                certency = np.max(prediction)

                # prediction = [1 if e >= self.threshold else 0 for e in prediction] # using threshold
                prediction = [1 if e == certency else 0 for e in prediction]

                target_class = list(target).index(1)
                target_class = list(classes.keys())[list(classes.values()).index(target_class)]

                try:
                    # catch, if no value is above the threshold (if used)
                    predicted_class = list(prediction).index(1)
                    predicted_class = list(classes.keys())[list(classes.values()).index(predicted_class)]
                except:
                    predicted_class = "unknowm"

                name = char_indices_to_string(char_indices=name)

                print("\n______________\n")
                print("name:", name)

                print("predicted as:", predicted_class, "(" + str(round(certency * 100, 4)) + "%)")
                print("actual target:", target_class)

        print("\ntest accuracy:", accuracy)


run = Run(model_file="models/model1.pt",
            dataset_path="datasets/final_matrix_name_list.pickle",
            epochs=5,
            # hyperparameters
            lr=0.0001,
            batch_size=512,
            threshold=0.4,
            hidden_size=256,
            layers=2,
            dropout_chance=0.65,
            embedding_size=128,
            continue_=True)

#run.train()
run.test()
