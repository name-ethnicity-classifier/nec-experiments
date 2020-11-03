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
from test_metrics import validate_accuracy, create_confusion_matrix, recall, precision, f1_score, score_plot

import xman


torch.manual_seed(0)


with open("datasets/preprocessed_datasets/final_nationality_to_number_dict.json", "r") as f: classes = json.load(f) 
total_classes = len(classes)


class Run:
    def __init__(self, model_file: str="", dataset_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=32, \
                    threshold: float=0.5, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5, bidirectional: bool=False, embedding_size: int=64, continue_: bool=False):
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
        self.bidirectional = bidirectional
        self.embedding_size = embedding_size

        self.continue_ = continue_

        # initialize experiment manager (uncomment if you have the xman libary installed)
        self.xmanager = xman.ExperimentManager(experiment_name="experiment8", new=False, continue_=self.continue_)
        self.xmanager.init(optimizer="RMSprop", 
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
                                "dense-layer-2": "logsoftmax"
                            })

    def _validate(self, model, dataset, confusion_matrix: bool=False, plot_scores: bool=False):
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
                target_index = targets[i].cpu().detach().numpy()[0]

                prediction = predictions[i].cpu().detach().numpy()
                prediction_index = list(prediction).index(max(prediction))

                total_targets.append(target_index)
                total_predictions.append(prediction_index)

        # calculate loss
        loss = np.mean(losses)

        # calculate accuracy
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=self.threshold)

        # calculate precision, recall and F1 scores
        precision_scores = precision(total_targets, total_predictions, classes=total_classes)
        recall_scores = recall(total_targets, total_predictions, classes=total_classes)
        f1_scores = f1_score(precision_scores, recall_scores)

        # create confusion matrix
        if confusion_matrix:
            create_confusion_matrix(total_targets, total_predictions, classes=classes)
        
        if plot_scores:
            score_plot(precision_scores, recall_scores, f1_scores, classes)

        return loss, accuracy, (precision_scores, recall_scores, f1_scores)

    def train(self):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance, bidirectional=self.bidirectional, embedding_size=self.embedding_size).to(device=device)
        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

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
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].cpu().detach().numpy()[0])
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))

            # scheduler.step()

            # calculate train loss and accuracy of last epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = validate_accuracy(total_train_targets, total_train_predictions, threshold=self.threshold)

            # calculate validation loss and accuracy of last epoch
            epoch_val_loss, epoch_val_accuracy, _ = self._validate(model, self.validation_set)

            # log training stats
            train_loss_history.append(epoch_train_loss); train_accuracy_history.append(epoch_train_accuracy)
            val_loss_history.append(epoch_val_loss); val_accuracy_history.append(epoch_val_accuracy)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)

            # save checkpoint of model
            torch.save(model.state_dict(), self.model_file)

            # log epoch results with xman (uncomment if you have the xman libary installed)
            self.xmanager.log_epoch(model, self.lr, self.batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

        # plot train-history with xman (uncomment if you have the xman libary installed)
        self.xmanager.plot_history(save=True)

    def test(self, print_: bool=True):
        model = Model(class_amount=total_classes, hidden_size=self.hidden_size, layers=self.layers, dropout_chance=self.dropout_chance, bidirectional=self.bidirectional, embedding_size=self.embedding_size).to(device=device)
        model.load_state_dict(torch.load(self.model_file))

        _, accuracy, scores = self._validate(model, self.test_set, confusion_matrix=True, plot_scores=True)

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

                prediction = [1 if e == certency else 0 for e in prediction]

                target_class = list(target).index(1)
                target_class = list(classes.keys())[list(classes.values()).index(target_class)]

                try:
                    # catch, if no value is above the threshold (if used)
                    predicted_class = list(prediction).index(1)
                    predicted_class = list(classes.keys())[list(classes.values()).index(predicted_class)]
                except:
                    predicted_class = "unknowm"

                if print_:
                    name = char_indices_to_string(char_indices=name)

                    print("\n______________\n")
                    print("name:", name)

                    print("predicted as:", predicted_class, "(" + str(round(certency * 100, 4)) + "%)")
                    print("actual target:", target_class)

        precisions, recalls, f1_scores = scores
        print("\n\ntest accuracy:", accuracy)
        print("\nprecision of every class:", precisions)
        print("\nrecall of every class:", recalls)
        print("\nf1-score of every class:", f1_scores)


run = Run(model_file="models/best_model8.pt",
            dataset_path="datasets/preprocessed_datasets/final_matrix_name_list.pickle",
            epochs=5,
            # hyperparameters
            lr=0.001,
            batch_size=512,
            threshold=0.4,
            hidden_size=256,
            layers=2,
            dropout_chance=0.6,
            bidirectional=False,
            embedding_size=128,
            continue_=False)


# run.train()
run.test(print_=True)
