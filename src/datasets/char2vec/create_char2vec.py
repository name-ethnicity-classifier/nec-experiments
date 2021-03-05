
from gensim.models import Word2Vec
import numpy as np
import string
import pickle5 as pickle
import json
import matplotlib.pyplot as plt
import re
from nltk import ngrams
from tqdm import tqdm


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


def create_n_gram(int_name: list, n: int) -> list:
    """ create n-gram sample from index representation

    :param list int_name: integer/index representation of the name
    :paran int n: gram size (n-gram)
    :return list: n-gram integer/index representation of the name
    """

    if n == 2:
        with open("../ngrams/bi_gram_table.json", "r") as b:
            n_gram_table = json.load(b)

    elif n == 3:
        with open("../ngrams/tri_gram_table.json", "r") as b:
            n_gram_table = json.load(b)
    else:
        raise ValueError("Only bi- or tri-gram possible!")

    str_name = ""
    for e in int_name:
        str_name += " " + str(e)
    
    sub_names = re.split(" 27 | 28 |27 | 27|28| 28", str_name)

    for s in range(len(sub_names)):
        sub_name = [l for l in sub_names[s].split(" ") if l != ""]
        sub_names[s] = [str(l) for l in sub_name]
        
    n_gram_name = []
    for i, sub_name in enumerate(sub_names):
        # n_gram_name += [(str(l[0]) + "$" + str(l[1])) for l in list(ngrams(sub_name, n))]
        n_gram_name += ["".join([("$" + str(l[i])) for i in range(len(l))])[1:] for l in list(ngrams(sub_name, n))]

        if i != len(sub_names) - 1:
            n_gram_name += ["27"]

    n_gram_name = [str(n_gram_table[l]) for l in n_gram_name]

    return n_gram_name


def preprocess(dataset: list, ngram: int=1) -> list:
    preprocessed_dataset = []
    for sample in tqdm(dataset, desc="creating {}-gram char2vec embedding".format(ngram), ncols=200):
        int_name = [str(e+1) for e in sample[1]]

        if ngram != 1:
            int_name = create_n_gram(int_name, n=ngram)
        
        preprocessed_dataset.append(int_name)

    return preprocessed_dataset


def char2vec(corpus: list=[], model_file: str="") -> None:
    model = Word2Vec(corpus, window=5, size=200)
    model.save(model_file)
    print("created char2vec model in '{}'.".format(model_file))


def load_gensim_model(model_file: str="") -> None:
    return Word2Vec.load(model_file)


def project_embedding(model_file: str="", n: int=1):
    np.random.seed(111)

    vocab_size = pow(26, n) + 1

    random_projection_matrix = np.random.rand(200, 2)
    
    xs, ys = [], []
    embedder = load_gensim_model(model_file=model_file)
    for i in range(vocab_size):
        try:
            embedding = embedder[str(i+1)]
            
            projection = list(np.dot(embedding, random_projection_matrix))
            xs.append(projection[0])
            ys.append(projection[1])
        except:
            pass

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, color="blue", alpha=0.6)

    alphabet = list(string.ascii_lowercase.strip()) + [" "]

    if n == 1:
        vocabulary = alphabet

    elif n == 2:
        with open("../ngrams/bi_gram_table.json", "r") as b:
            n_gram_table = json.load(b)

    elif n == 3:
        with open("../ngrams/tri_gram_table.json", "r") as b:
            n_gram_table = json.load(b)

    if n == 2 or n == 3:
        vocabulary = []
        for letter_idx_pairs in n_gram_table:
            letter_pair = "".join([alphabet[int(idx)-1] for idx in letter_idx_pairs.split("$")])
            vocabulary.append(letter_pair)

    for i in range(len(xs)):
        ax.annotate(vocabulary[i], (xs[i], ys[i]))
    plt.show()



if __name__ == "__main__":
    with open("../preprocessed_datasets/final_more_nationalities/matrix_name_list.pickle", "rb") as f:
        dataset = pickle.load(f)
        print(len(dataset))
        print(len(dataset) / 23)
    # train embeddings only on train-dataset, split in same way as in the normal training
    test_size = int(np.round(len(dataset)*0.025))
    print(test_size)
    val_size = int(np.round(len(dataset)*0.025))
    print(val_size)
    dataset = dataset[(test_size+val_size):]
    print(len(dataset))
    
    """ uni-gram char2vec 
    corpus = preprocess(dataset)
    char2vec(corpus=corpus, model_file="./gensim_unigram_model.model")
    embedder = load_gensim_model(model_file="./gensim_unigram_model.model")
    project_embedding(model_file="./gensim_unigram_model.model")
    """

    """ bi-gram char2vec
    corpus = preprocess(dataset, ngram=2)
    char2vec(corpus=corpus, model_file="./gensim_bigram_model.model")
    embedder = load_gensim_model(model_file="./gensim_bigram_model.model")
    project_embedding(model_file="./gensim_bigram_model.model", n=2)
    """

    """ tri-gram char2vec 
    corpus = preprocess(dataset, ngram=3)
    char2vec(corpus=corpus, model_file="./gensim_trigram_model.model")
    embedder = load_gensim_model(model_file="./gensim_trigram_model.model")
    project_embedding(model_file="./gensim_trigram_model.model", n=3)
    """

