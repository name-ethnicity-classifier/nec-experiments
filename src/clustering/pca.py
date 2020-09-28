
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


with open("../datasets/final_nationality_to_number_dict.json", "r") as f: 
    classes = json.load(f)
    
total_classes = len(classes)


def create_projections(dataset: list=[]) -> list:
    # remove labels
    data = []
    for sample in dataset:
        data.append(sample[0])

    # create PCA 2D projection of the dataset
    pca = PCA(3)
    projected_data = pca.fit_transform(data)

    # add labels to the projections
    final_projections = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        final_projections.append([projected_data[i].tolist(), target])

    return final_projections


def create_clusters(dataset_path: str=""):
    dataset = np.load(dataset_path, allow_pickle=True)
    projections = create_projections(dataset=dataset)

    class_clusters = [[] for _ in range(total_classes)]
    for sample in projections:
        coordinate, target = sample[0], sample[1][0]
        class_clusters[target].append(coordinate)

    colors = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:gray", "orange", "m"]

    fig = plt.figure()
    ax = Axes3D(fig)

    amount = 500
    for i, cluster in enumerate(class_clusters):
        cluster = list(zip(*cluster))
        ax.scatter(cluster[0][:amount], cluster[1][:amount], cluster[2][:amount], color=colors[i], label=list(classes.keys())[list(classes.values()).index(i)])
    
    ax.legend()
    plt.show()

    def rotate(angle):
        ax.view_init(azim=angle)

    rotation_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    rotation_animation.save("results/pca_rotation.gif", dpi=80, writer="imagemagick")



create_clusters(dataset_path="dataset/lstm_embeddings_test.npy")