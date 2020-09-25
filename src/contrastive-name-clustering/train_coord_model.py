

""" file to train and evaluate the model """

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

import torch
import torch.utils.data
import torch.nn as nn

from coord_model import CoordinateModel as Model
from utils import create_dataloader, show_progress, device
from contrastive_loss import ContrastiveCosineLoss

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation



with open("../datasets/final_nationality_to_number_dict.json", "r") as f: classes = json.load(f) 
total_classes = len(classes)


class Run:
    def __init__(self, model_file: str="", dataset_path: str="", epochs: int=10, lr: float=0.001, batch_size: int=32, input_size: int=256, \
                    dropout_chance: float=0.5, continue_: bool=False):

        self.model_file = model_file
        self.dataset_path = dataset_path
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=0.025, val_size=0.025, batch_size=batch_size, class_amount=total_classes)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.input_size = input_size
        self.dropout_chance = dropout_chance

        self.continue_ = continue_

    def _create_targets(self, coordinates, targets):
        partner_coordinates = coordinates.flip(-2)
        partner_targets = targets.flip(-2)

        class_differences = []
        for idx in range(coordinates.shape[0]):
            target = targets[idx].cpu().detach().numpy()
            partner_target = partner_targets[idx].cpu().detach().numpy()

            if target == partner_target:
                class_differences.append(0)
            else:
                class_differences.append(1)

        return partner_coordinates, torch.tensor(class_differences).to(device=device)

    def train(self):
        model = Model(input_size=self.input_size, dropout_chance=self.dropout_chance).to(device=device)
        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = ContrastiveCosineLoss(margin=0.0, eps=1e-08, beta=0.75)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        loss_history = []
        for epoch in range(1, (self.epochs + 1)):

            epoch_loss = []
            for inputs, targets in tqdm(self.train_set, desc="epoch", ncols=150):
                optimizer.zero_grad()

                inputs, targets = inputs.to(device=device), targets.to(device=device)

                coordinates = model.train()(inputs)
                partner_coordinates, class_differences = self._create_targets(coordinates, targets)

                loss = criterion(coordinates, partner_coordinates, class_differences)
                loss.backward()
                optimizer.step()

                # log train loss
                epoch_loss.append(loss.item())
                
            current_epoch_loss = np.mean(epoch_loss)
            loss_history.append(current_epoch_loss)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, current_epoch_loss, 0, 0, 0)

            # save checkpoint of model
            torch.save(model.state_dict(), self.model_file)

    def test(self):
        model = Model(input_size=self.input_size, dropout_chance=self.dropout_chance).to(device=device)
        model.load_state_dict(torch.load(self.model_file))

        predicted_class_clusters = [[] for _ in range(total_classes)]
        for inputs, targets in tqdm(self.train_set, desc="creating clusters", ncols=150):

            inputs, targets = inputs.to(device=device), targets.to(device=device)
            coordinates = model.eval()(inputs)
            
            for idx in range(inputs.shape[0]):
                coordinate = coordinates[idx].cpu().detach().numpy()
                target = int(targets[idx].cpu().detach().numpy()[0])

                predicted_class_clusters[target].append(coordinate)

        colors = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:gray", "tab:brown", "m"]
        random.shuffle(colors)

        fig = plt.figure()
        ax = Axes3D(fig)



        amount = 500
        for i, cluster in enumerate(predicted_class_clusters):
            cluster = list(zip(*cluster))
            ax.scatter(cluster[0][:amount], cluster[1][:amount], cluster[2][:amount], color=colors[i], label=list(classes.keys())[list(classes.values()).index(i)])
        
        ax.legend()
        # plt.show()
        def rotate(angle):
            ax.view_init(azim=angle)

        rotation_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
        rotation_animation.save('rotation2.gif', dpi=80, writer='imagemagick')


run = Run(model_file="coord_models/coord_model2.pt",
            dataset_path="lstm_embeddings.npy",
            epochs=200,
            # hyperparameters
            lr=0.0005,
            batch_size=1024,
            input_size=32,
            dropout_chance=0.15,
            continue_=False)


# run.train()
run.test()

"""d = np.load("lstm_embeddings.npy", allow_pickle=True)
for i in d:
    print(i)"""

