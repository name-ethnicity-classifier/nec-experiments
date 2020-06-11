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
from utils import create_dataloader, validate_accuracy, show_progress, onehot_to_string, init_xavier_weights, plot, device


torch.manual_seed(0)


with open("datasets/nationality_to_number_dict.json", "r") as f: classes = json.load(f) 
total_classes = len(classes)


class Run:
    def __init__(self, model_file: str="", dataset_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=32, threshold: float=0.5, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5):
        self.model_file = model_file
        self.dataset_path = dataset_path
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=0.01, val_size=0.01, batch_size=batch_size)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.threshold = threshold

        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

    def _validate(self, model, dataset):
        validation_dataset = dataset

        total_targets, total_predictions = [], []

        for names, targets, _ in tqdm(validation_dataset, desc="validating", ncols=150):
            names, targets = names.to(device=device), targets.to(device=device)

            predictions = model.eval()(names, len(names[0]), len(names))

            for i in range(predictions.size()[0]):
                total_targets.append(targets[i].cpu().detach().numpy())
                total_predictions.append(predictions[i].cpu().detach().numpy())

        # calculate accuracy
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=self.threshold)

        # calculate loss
        criterion = nn.NLLLoss()
        loss = criterion(predictions, targets.squeeze())
        loss = np.mean(loss.cpu().detach().numpy())
        return loss, accuracy

    def train(self, continue_: bool=False):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance).to(device=device)

        if continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

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
                validated_predictions = predictions
                for i in range(validated_predictions.size()[0]): total_train_targets.append(targets[i].cpu().detach().numpy()); \
                                                        total_train_predictions.append(validated_predictions[i].cpu().detach().numpy())

            # decay learning rate (if wanted, uncomment)
            """if epoch % 3 == 0:
                self.lr = self.lr / 2"""

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

        plot(train_accuracy_history, train_loss_history, val_accuracy_history, val_loss_history)

    def test(self):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance).to(device=device)
        model.load_state_dict(torch.load(self.model_file))

        _, accuracy = self._validate(model, self.test_set)

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
                prediction = list(prediction).index(max(prediction))
                prediction_empty = np.zeros((amount_classes))
                prediction_empty[prediction] = 1
                prediction = prediction_empty

                target_class = list(target).index(1)
                predicted_class = list(prediction).index(1)

                target_class = list(classes.keys())[list(classes.values()).index(target_class)]
                predicted_class = list(classes.keys())[list(classes.values()).index(predicted_class)]

                print("\n______________\n")
                print("name:", onehot_to_string(non_padded_name))
                print("predicted as:", predicted_class)
                print("actual target:", target_class)

        print("\ntest accuracy:", accuracy)


run = Run(model_file="models/model5.pt",
            dataset_path="datasets/matrix_name_list_4000.pickle",
            epochs=100,
            # hyperparameters
            lr=0.0001,
            batch_size=128,
            threshold=0.5,
            hidden_size=256,
            layers=2,
            dropout_chance=0.5)

run.train(continue_=False)
run.test()


# model1: rnn, relu
# model2: rnn, relu
# model3: gru, sigmoid
# model4: rnn, tanh, 4l
# model5: rnn, tanh
