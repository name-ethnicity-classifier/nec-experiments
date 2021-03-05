
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import re
from nltk import ngrams
from tqdm import tqdm


with open("../preprocessed_datasets/index_final_matrix_name_list.pickle", "rb") as f:
    dataset = pickle.load(f)

with open("bi_gram_table.json", "r") as f:
    bi_gram_table = json.load(f)

with open("tri_gram_table.json", "r") as f:
    tri_gram_table = json.load(f)

with open("quad_gram_table.json", "r") as f:
    quad_gram_table = json.load(f)

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",",_,"]
bi_letter_pairs = list(bi_gram_table.keys())
tri_letter_pairs = list(tri_gram_table.keys())
quad_letter_pairs = list(quad_gram_table.keys())


def create_n_gram(int_name: list, n_gram_table: dict, n: int=2) -> list:
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
        # print(n_gram_name)
        if i != len(sub_names) - 1:
            n_gram_name += ["27"]

    n_gram_name = [n_gram_table[l] for l in n_gram_name]

    return n_gram_name


def count_letter_pairs(letter_pairs: list, n_gram_table: dict, n: int=1):
    n_counter_list = [0 for _ in range(len(letter_pairs))]
    uni_counter_list = [0 for _ in range(27)]

    for idx_name in tqdm(dataset, desc="counting pairs", ncols=150):
        int_name, label = idx_name[1], idx_name[0]
        int_name = [e+1 for e in int_name]
        
        for letter_idx in int_name:
            if letter_idx == 28:
                letter_idx = 27
        
            uni_counter_list[letter_idx-1] += 1

        n_idx_name = create_n_gram(int_name, n_gram_table, n=n)
        for letter_pair_idx in n_idx_name:
            n_counter_list[letter_pair_idx-1] += 1

    return n_counter_list, uni_counter_list


def plot_distrubutions(letter_pairs: list, uni_counter_list: list, n_counter_list: list, n_gram_table: dict, print_head: tuple=(True, 5)):
    global alphabet

    zipped_uni_data = sorted(list(zip(uni_counter_list, alphabet)))[::-1]
    uni_counter_list, alphabet = list(zip(*zipped_uni_data))

    zipped_bi_data = sorted(list(zip(n_counter_list, letter_pairs)))[::-1]
    n_counter_list, letter_pairs = list(zip(*zipped_bi_data))

    if print_head[0] == True and print_head[1] > 0:
        original_letter_pairs = []
        most_frequent = letter_pairs[:print_head[1]]

        for letter_pair in most_frequent:
            if letter_pair == "27":
                original_letter_pairs.append("<space>")
            else:
                letter_indices = [int(idx) for idx in letter_pair.split("$")]

                letter_pair = "".join([alphabet[idx-1] for idx in letter_indices])
                original_letter_pairs.append(letter_pair)

        print(original_letter_pairs)

    #plt.style.use("bmh")
    """fig, axs = plt.subplots(2, 1)
    axs[0].barh(alphabet, uni_counter_list)
    axs[1].barh(list(range(len(letter_pairs))), n_counter_list)
    plt.show()

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(np.log(list(range(len(alphabet)))), np.log(uni_counter_list))
    axs[1].scatter(np.log(list(range(len(letter_pairs)))), np.log(n_counter_list))
    plt.show()"""



#bi_counter_list, uni_counter_list = count_letter_pairs(bi_letter_pairs, bi_gram_table, n=2)
#plot_distrubutions(bi_letter_pairs, uni_counter_list, bi_counter_list, bi_gram_table, print_head=(True, 7))


#tri_counter_list, uni_counter_list = count_letter_pairs(tri_letter_pairs, tri_gram_table, n=3)
#plot_distrubutions(tri_letter_pairs, uni_counter_list, tri_counter_list, tri_gram_table, print_head=(True, 7))


quad_counter_list, uni_counter_list = count_letter_pairs(quad_letter_pairs, quad_gram_table, n=4)
plot_distrubutions(quad_letter_pairs, uni_counter_list, quad_counter_list, quad_gram_table, print_head=(True, 7))