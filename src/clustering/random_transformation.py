
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from utils import device, create_dataloader


with open("../datasets/final_nationality_to_number_dict.json", "r") as f: 
    classes = json.load(f)
    
total_classes = len(classes)


class RandomTransformer(nn.Module):
    def __init__(self, input_size: int=256):
        super(RandomTransformer, self).__init__()

        self.input_size = input_size
        self.linear = nn.Linear(self.input_size, 64)
        
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))

        return x


def create_clusters(dataset_path: str="", embedding_size: int=256, batch_size=512):
    randomTransform = RandomTransformer(input_size=embedding_size).to(device=device)
    dataset = create_dataloader(dataset_path=dataset_path, batch_size=batch_size, class_amount=total_classes)

    class_clusters = [[] for _ in range(total_classes)]
    for inputs, targets in tqdm(dataset, desc="creating clusters", ncols=150):

        inputs, targets = inputs.to(device=device), targets.to(device=device)
        # matrix mulitplication with random matrix
        coordinates = randomTransform.eval()(inputs)
        
        for idx in range(inputs.shape[0]):
            coordinate = coordinates[idx].cpu().detach().numpy()
            target = int(targets[idx].cpu().detach().numpy()[0])

            class_clusters[target].append(coordinate)

    colors = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:gray", "orange", "m"]

    fig = plt.figure()
    ax = Axes3D(fig)

    amount = 500
    for i, cluster in enumerate(class_clusters):
        cluster = list(zip(*cluster))
        ax.scatter(cluster[0][:amount], cluster[1][:amount], cluster[2][:amount], color=colors[i], label=list(classes.keys())[list(classes.values()).index(i)])
    
    ax.legend()
    plt.title("random transformation")
    plt.show()

    def rotate(angle):
        ax.view_init(azim=angle)

    rotation_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    rotation_animation.save("rt_rotation.gif", dpi=80, writer="imagemagick")


create_clusters(dataset_path="dataset/lstm_embeddings_test.npy", embedding_size=256, batch_size=1024)