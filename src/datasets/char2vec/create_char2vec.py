
from gensim.models import Word2Vec
import numpy as np
import string
import pickle5 as pickle
import json
import matplotlib.pyplot as plt

plt.style.use("bmh")

def char_indices_to_string(char_indices: list=[str]) -> str:
    """ takes a list with indices from 0 - 27 (alphabet + " " + "-") and converts them to a string

        :param str char_indices: list containing the indices of the chars
        :return str: decoded name
    """

    alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]
    name = ""
    for idx in char_indices:
        if int(idx) == 0:
            pass
        else:
            name += alphabet[int(idx) - 1]
    
    return name


def preprocess(dataset: list) -> list:
    preprocessed_dataset = []
    for sample in dataset:
        int_name = [str(e+1) for e in sample[1]]
        preprocessed_dataset.append(int_name)

    return preprocessed_dataset


def char2vec(corpus: list=[], model_file: str="") -> None:
    model = Word2Vec(corpus, window=3, size=200)
    model.save(model_file)
    print("created char2vec model in '{}'.".format(model_file))


def load_gensim_model(model_file: str="") -> None:
    return Word2Vec.load(model_file)


def project_embedding(model_file: str=""):
    #np.random.seed(1)
    random_projection_matrix = np.random.rand(200, 2)
    
    fig, ax = plt.subplots()

    xs, ys = [], []

    embedder = load_gensim_model(model_file="./gensim_model.model")
    alphabet = list(string.ascii_lowercase.strip()) + [" "]
    for i in range(len(alphabet)):
        embedding = embedder[str(i+1)]
        
        projection = list(np.dot(embedding, random_projection_matrix))
        xs.append(projection[0])
        ys.append(projection[1])

    ax.scatter(xs, ys, color="blue", alpha=0.6)

    for i in range(len(xs)):
        ax.annotate(alphabet[i], (xs[i], ys[i]))
    plt.show()



if __name__ == "__main__":
    with open("../preprocessed_datasets/index_final_matrix_name_list.pickle", "rb") as f:
        dataset = pickle.load(f)

    corpus = preprocess(dataset)
    char2vec(corpus=corpus, model_file="./gensim_model.model")

    embedder = load_gensim_model(model_file="./gensim_model.model")

    project_embedding(model_file="./gensim_model.model")

