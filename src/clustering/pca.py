
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import load_json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.patches as mpatches

style.use("ggplot")


def create_projections(dataset: list, dimensions: int=3) -> list:
    # remove labels
    data = []
    for sample in dataset:
        data.append(sample[0])

    # create PCA projection of the dataset
    pca = PCA(dimensions)
    projected_data = pca.fit_transform(data)

    # add labels to the projections
    final_projections = []
    for i in range(len(dataset)):
        final_projections.append([projected_data[i].tolist(), dataset[i][1], dataset[i][2], dataset[i][3]])

    return final_projections


def plot_clusters_2d(class_clusters: list, classes: dict, point_size: int=20, amount: int=500) -> None:
    colors = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:gray", "orange", "m"]

    for i, cluster in enumerate(class_clusters):
        #genders = gender_clusters[i][:amount]
        #edge_colors = ["r" if gender == 1 else "b" for gender in genders]

        cluster = list(zip(*cluster))
        plt.scatter(cluster[0][:amount], cluster[1][:amount], s=point_size, color=colors[i], label=list(classes.keys())[list(classes.values()).index(i)])
    
    plt.show()
    

def plot_clusters_3d(class_clusters: list, classes: dict, point_size: int=20, amount: int=500) -> None:
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:gray", "orange", "m"]

    for i, cluster in enumerate(class_clusters):
        cluster = list(zip(*cluster))
        ax.scatter(cluster[0][:amount], cluster[1][:amount], cluster[2][:amount], s=point_size, color=colors[i], label=list(classes.keys())[list(classes.values()).index(i)])
    
    ax.legend()
    plt.show()

    def rotate(angle):
        ax.view_init(azim=angle)

    rotation_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)


def plot_genders_2d(class_clusters: list, gender_information: list, classes: dict, point_size: int=20, amount: int=1000) -> None:
    grid_dimension = int(np.sqrt(len(classes))) + 1
    fig, axs =  plt.subplots(grid_dimension, grid_dimension)

    c, i, j = 0, 0, 0
    while True:
        if i == grid_dimension:
            j += 1; i = 0

        if i == grid_dimension - 1 and j == grid_dimension - 1:
            axs[i][j].set_visible(False)
            break            
        
        cluster = class_clusters[c]
        genders = gender_information[c][:amount]
        colors = ["orange" if gender == 1 else "blue" for gender in genders]
        labels = ["female" if gender == 1 else "male" for gender in genders]
        alphas = [0.65 if gender == 1 else 0.35 for gender in genders]

        cluster = list(zip(*cluster))
        axs[i][j].scatter(cluster[0][:amount], cluster[1][:amount], s=point_size, color=colors, label="feamle", alpha=alphas)
        axs[i][j].set_title(list(classes.keys())[list(classes.values()).index(c)])

        i += 1; c += 1

    plt.subplots_adjust(hspace=0.5)

    fig.legend(handles=[mpatches.Patch(color="orange", label="female"), mpatches.Patch(color="blue", label="male")])
    plt.show()


def create_clusters(dataset_file: str, classes_file: str, point_size: int=20, dimensions: int=3, plot_genders: bool=False, amount: int=500) -> None:
    # make sure it's a 2d plot for the gender graphic
    if plot_genders:
        dimensions = 2

    # load embeddings and nationality-classes
    classes = load_json(classes_file)
    dataset = np.load(dataset_file, allow_pickle=True)

    if amount > len(dataset):
        amount = len(dataset)

    # create PCA projections
    projections = create_projections(dataset, dimensions=dimensions)

    # split sample projections into classes
    gender_clusters = [[] for _ in range(len(classes))]
    class_clusters = [[] for _ in range(len(classes))]
    for sample in projections:
        coordinate, target = sample[0], sample[1][0]
        class_clusters[target].append(coordinate)
        gender_clusters[target].append(sample[2])

    # plot genders
    if plot_genders:
        plot_genders_2d(class_clusters, gender_clusters, classes, point_size=point_size, amount=amount)
    # create 3d plot
    elif dimensions == 3:
        plot_clusters_3d(class_clusters, classes, point_size=point_size, amount=amount)

    # create 2d plot
    elif dimensions == 2:
        plot_clusters_2d(class_clusters, classes, point_size=point_size, amount=amount)



if __name__ == "__main__":
    embedding_folder = "8_groups"
    create_clusters(f"embeddings/{embedding_folder}/embeddings.npy", 
                    f"embeddings/{embedding_folder}/nationalities.json",
                     point_size=15, 
                     dimensions=3, 
                     plot_genders=False, 
                     amount=400)