import pickle
import random
import os
import json
from tqdm import tqdm
import argparse


def get_matrix_from_name(name: str, abc_dict: list):
    matrix = []
    for letter in name:
        matrix.append(abc_dict[letter])
    return matrix

def get_name_from_matrix(matrix: list, abc_list: list):
    name = ""
    for letter in matrix:
        index = letter
        letter = abc_list[index]
        name += letter
    return name

def handle_clusters(nationality: str, dict_clusters: dict):
    for key in dict_clusters:
        if nationality in dict_clusters[key]:
            return key
    return 'other'

def max_per_cluster(cluster_dict: dict, amount_names_country: dict):
    max_per_cluster = {}
    for key in cluster_dict:

        smallest = 1000000000000
        for country in cluster_dict[key]:

            if country in amount_names_country:
                if amount_names_country[country] <= smallest:
                    smallest = amount_names_country[country]

        for country in cluster_dict[key]:
            max_per_cluster[country] = smallest

    return max_per_cluster


def preprocess(dataset_name: str="", nationalities: str="", raw_dataset_path: str=""):
    # load raw dataset
    with open(raw_dataset_path, "rb") as o:
        dict_chosen_names = pickle.load(o)

    # set lower limit of names per country if wanted
    minimum_per_country = 1

    # abc_dict is a dictionary where the letters "a"-"z" and " " and "-" are keys to lists representing these values in the matrix_name_list
    abc_dict = {}
    abc_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ","-"]
    a = list(range(len(abc_list)))
    for i in range(len(abc_list)):
        abc_dict[abc_list[i]] = a[i]

    # get name amount per country
    amount_names_chountry = {}
    dict_chosen_names_2 = dict_chosen_names.copy()
    for key in dict_chosen_names_2:

        if len(dict_chosen_names[key]) <= minimum_per_country:
            dict_chosen_names.pop(key)
        else:
            amount_names_chountry[key]= len(dict_chosen_names[key])

    # create nationality selection dictionary
    all_nationalities = ['british', 'indian', 'american', 'german', 'polish', 'pakistani', 'italian', 'romanian', 'french', 'chinese', 'irish', 'japanese', 'spanish', 'filipino', 'dutch', 'nigerian', 'south korean', 'taiwanese', 'hong konger','korean', 'swiss', 'danish', 'austrian','belgian', 'luxembourger', 'portugese',
                        'norwegian', 'swedish', 'finnish', 'icelandic', 'denmark','lithuanian', 'estonian', 'latvian', 'hungarian', 'bulgarian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian',
                        'cypriot', 'greek', 'russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh','kyrgyz', 
                        'nepalese', 'sri lankan', 'singaporean','bangladeshi', 'malaysian', 'fijian', 'thai', 'indonesian', 'burmese', 'vietnamese',
                        'turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian','maltese',
                        'egyptian', 'moroccan', 'tunisian','libyan', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan',
                        'zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan', 'namibian','citizen of seychelles',
                        'canadian', 'new zealander','mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander', 'costa rican','grenadian','panamanian', 'cuban',
                        'brazilian', 'colombian', 'argentinian', 'peruvian', 'venezuelan', 'ecuadorean','chilean', 'guyanese','bolivian','uruguayan']
        
    chosen_nationalities_dict = {}
    available_nationalities = all_nationalities.copy()
    for nationality in nationalities:
        if nationality == "else":
            continue

        chosen_nationalities_dict[nationality] = [nationality]
        available_nationalities.pop(available_nationalities.index(nationality))
        
    if "else" in nationalities:
        chosen_nationalities_dict["else"] = available_nationalities

    # gather equally distributed names of all chosen countries
    max_per_cluster_dict = max_per_cluster(chosen_nationalities_dict, amount_names_chountry)
    matrix_name_dict = {}
    nationality_to_number_dict = {}
    number = 0

    for key in tqdm(dict_chosen_names):
        try:

            max_nat = max_per_cluster_dict[key]
            counter = 0

            list_of_names = dict_chosen_names[key]
            random.shuffle(list_of_names)

            all_names = []
            for name in list_of_names:
                name = name.lower()

                # remove "dr", "ms", "mr", "mrs"
                if name.split(" ")[0] == "dr" or name.split(" ")[0] == "mr" or name.split(" ")[0] == "ms" or name.split(" ")[0] == "miss" or name.split(" ")[0] == "mrs":
                    space_idx = name.strip().index(" ")
                    name = name[space_idx:]

                # remove random spaces before name
                if list(name)[0] == " ":
                    name = name[1:]

                if counter <= max_nat:
                    name = name.strip()
                    nationality = key

                    org_nat = nationality
                    nationality = handle_clusters(nationality, chosen_nationalities_dict)

                    if nationality not in nationality_to_number_dict and nationality != 'other':
                        nationality_to_number_dict[nationality] = number
                        number += 1
                        matrix_name_dict[nationality_to_number_dict[nationality]]=[get_matrix_from_name(name, abc_dict)]
                    elif nationality in nationality_to_number_dict and nationality != 'other':
                        matrix_name_dict[nationality_to_number_dict[nationality]]+=[get_matrix_from_name(name, abc_dict)]       
                    counter += 1
                else:
                    name = name.strip()
                    nationality = key
                    org_nat = nationality
                    nationality = handle_clusters(nationality, chosen_nationalities_dict)

                    if nationality not in nationality_to_number_dict and nationality!= 'other':
                        nationality_to_number_dict[nationality]= number
                        number += 1
                        matrix_name_dict[nationality_to_number_dict[nationality]]=[get_matrix_from_name(name, abc_dict)]
                    elif nationality in nationality_to_number_dict and nationality!= 'other':
                            matrix_name_dict[nationality_to_number_dict[nationality]]+=[get_matrix_from_name(name, abc_dict)]
        except:
            pass

    matrix_name_list = []
    nr_of_countries = 0
    list_countries_used = []

    minimum_per_country = min([len(matrix_name_dict[country]) for country in matrix_name_dict])
    for country in matrix_name_dict:

        if len(matrix_name_dict[country]) >= minimum_per_country:
            list_countries_used += [country]
            nr_of_countries += 1
            names = matrix_name_dict[country]
            random.shuffle(names)
            names = names[:minimum_per_country]

            for name in names:
                matrix_name_list += [[nr_of_countries, name]]

    random.shuffle(matrix_name_list)

    """ SAVE DATASET FILES """

    dataset_path = "datasets/preprocessed_datasets/" + dataset_name
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with open(dataset_path + "/dataset.pickle", "wb+") as o:
        pickle.dump(matrix_name_list, o, pickle.HIGHEST_PROTOCOL)

    names_countries_used = {}
    for i, element in enumerate(list_countries_used):
        country_name = list(nationality_to_number_dict.keys())[list(nationality_to_number_dict.values()).index(element)]
        names_countries_used[country_name] = i

    filepath = dataset_path + "/nationalities.json"
    with open(filepath, 'w+') as f:
        json.dump(names_countries_used, f, indent=4)



preprocess(dataset_name="test-dataset", nationalities=["british", "else", "pakistani", "german", "chinese", "spanish", "italian", "dutch"], raw_dataset_path="datasets/raw_datasets/total_names_dataset.pickle")