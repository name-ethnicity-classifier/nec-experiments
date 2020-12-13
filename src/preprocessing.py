import pickle
import random
import os
import json
from tqdm import tqdm
#the higher the variable number_of_names_per_country, the less nationalities are represented in the output matrix and the more names are given per nationality

#recommended number_of_names_per_country:
#level_medium_clusters: --> 5.500
#level_big_clusters: --> 42.000
number_of_names_per_country = 6000

#only one of these boolean values should be true, it decides if the nationalities should be clusterred and in which way they should be clustered
level_nationality = True
level_medium_clusters = False
level_big_clusters = False
experimental = False
minimum_per_country = 277
with open("datasets/raw_datasets/total_names_dataset.pickle", "rb") as o:
    dict_chosen_names = pickle.load(o)

#abc_dict is a dictionary where the letters "a"-"z" and " " and "-" are keys to lists representing these values in the matrix_name_list
abc_dict = {}
abc_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ","-"]
# a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
a = list(range(len(abc_list)))
for i in range(len(abc_list)):
    abc_dict[abc_list[i]] = a[i]#a[-i:]+a[:-i]


def get_matrix_from_name(name,abc_dict):
    # sub_names = name.split(" ")
    # name = sub_names[0]
    matrix = []
    for letter in name:
        matrix.append(abc_dict[letter])
    return matrix

def get_name_from_matrix(matrix,abc_list):
    name = ""
    for letter in matrix:
        index = letter#.index(1)
        letter = abc_list[index]
        name += letter
    return name

amount_names_chountry = {}
dict_chosen_names_2 = dict_chosen_names.copy()
for key in dict_chosen_names_2:

    if len(dict_chosen_names[key]) <= minimum_per_country:
        dict_chosen_names.pop(key)
    else:
        amount_names_chountry[key]= len(dict_chosen_names[key])
        #print("in main names ",key)


big_clusters_dict = {'european':['british', 'irish', 'german', 'dutch', 'swiss', 'danish', 'austrian','belgian', 'french', 'luxembourger', 'spanish', 'portugese', 'italian', 'romanian',
                    'norwegian', 'swedish', 'finnish', 'icelandic', 'denmark','lithuanian', 'estonian', 'latvian', 'hungarian', 'polish', 'bulgarian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian',
                     'cypriot', 'greek'],

                    'asian':['russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh','kyrgyz', 'chinese', 'japanese', 'south korean', 'taiwanese', 'hong konger','korean', 
                    'indian', 'pakistani', 'nepalese', 'sri lankan', 'singaporean','bangladeshi', 'malaysian', 'fijian', 'thai', 'filipino', 'indonesian', 'burmese', 'vietnamese'],
                    
                    'african':['turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian','maltese',
                    'egyptian', 'moroccan', 'tunisian','libyan', 'nigerian', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan',
                    'zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan', 'namibian','citizen of seychelles'],  
                    
                    'american':['canadian', 'american', 'new zealander','australian','mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander', 'costa rican','grenadian','panamanian', 'cuban',
                    'brazilian', 'colombian', 'argentinian', 'peruvian', 'venezuelan', 'ecuadorean','chilean', 'guyanese','bolivian','uruguayan']}
medium_clusters_dict = {'anglo_european':['british'],

                        'germanic':['german', 'dutch', 'swiss', 'danish', 'austrian'],

                        'frankophone':['belgian', 'french', 'luxembourger'],

                        'romanic':['spanish', 'portugese', 'italian', 'romanian'],

                        'scandinavian':['norwegian', 'swedish', 'finnish', 'icelandic', 'denmark'],

                        'slavic':['lithuanian', 'estonian', 'latvian', 'hungarian', 'polish', 'bulgarian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian'],

                        'greco':['cypriot', 'greek'],
                        
                        'anglo_american':['canadian', 'american', 'new zealander','australian'],
                        
                        'north_asian':['russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh','kyrgyz'],
                        
                        'central_asian':['chinese', 'japanese', 'south korean', 'taiwanese', 'hong konger','korean'],
                        
                        'south_asian':['indian', 'pakistani', 'nepalese', 'sri lankan', 'singaporean'],
                        
                        'east_asian':['bangladeshi', 'malaysian', 'fijian', 'thai', 'filipino', 'indonesian', 'burmese', 'vietnamese'],
                        
                        'middle_eastern':['turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian','maltese'],
                        
                        'north_african':['egyptian', 'moroccan', 'tunisian','libyan'],
                        
                        'central_african':['nigerian', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan'],
                        
                        'south_african':['zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan', 'namibian','citizen of seychelles'],
                        
                        'central_american':['mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander', 'costa rican','grenadian','panamanian', 'cuban'],
                        
                        'south_american':['brazilian', 'colombian', 'argentinian', 'peruvian', 'venezuelan', 'ecuadorean','chilean', 'guyanese','bolivian','uruguayan']}

experimental_dict = {'british':['british'],
                    'indian':['indian'],
                    'german':['german'],
                    'polish':['polish'],
                    'russian':['russian'],
                    'italian':['italian'],
                    'romanian':['romanian'],
                    'french':['french'],
                    'chinese':['chinese'],
                    'japanese':['japanese'],
                    'spanish':['spanish'],
                    'hungarian':['hungarian'],
                    'greek':['greek'],
                    'danish':['danish'],
                    'zimbabwean':['zimbabwean'],
                    'turkish':['turkish'],
                    'pakistani':['pakistani'],
                    'south korean':['south korean'],
                    'czech':['check'],
                    'bulgarian':['bulgarian'],
                    'israeli':['israeli'],
                    'swedish': ['swedish'],
                    'dutch':['dutch'],
                    'ukrainian':['ukrainian'],
                    'nigerian':['nigerian'],
                    'else':['irish', 'filipino', 'taiwanese', 'hong konger', 'swiss', 'austrian', 'luxembourger', 'portugese',
                    'norwegian', 'finnish', 'icelandic', 'denmark','lithuanian', 'estonian', 'latvian', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian',
                    'cypriot', 'uzbek', 'moldovan', 'turkmen','kazakh','kyrgyz', 
                    'nepalese', 'sri lankan', 'singaporean','bangladeshi', 'malaysian', 'fijian', 'thai', 'indonesian', 'burmese', 'vietnamese',
                    'iraqi', 'iranian', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian','maltese',
                    'egyptian', 'moroccan', 'tunisian','libyan', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan',
                    'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan', 'namibian','citizen of seychelles',
                    'canadian', 'mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander', 'costa rican','grenadian','panamanian', 'cuban',
                    'brazilian', 'colombian', 'argentinian', 'peruvian', 'venezuelan', 'ecuadorean','chilean', 'guyanese','bolivian','uruguayan']}

""" original
experimental_dict = {'british':['british'],
                    'indian':['indian'],
                    'american':['american'],
                    'german':['german'],
                    'polish':['polish'],
                    'pakistani':['pakistani'],
                    'italian':['italian'],
                    'romanian':['romanian'],
                    'french':['french'],
                    'chinese':['chinese'],
                    'else':['irish', 'japanese', 'spanish', 'filipino', 'dutch', 'nigerian', 'south korean', 'taiwanese', 'hong konger','korean', 'swiss', 'danish', 'austrian','belgian', 'luxembourger', 'portugese',
                    'norwegian', 'swedish', 'finnish', 'icelandic', 'denmark','lithuanian', 'estonian', 'latvian', 'hungarian', 'bulgarian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian',
                    'cypriot', 'greek', 'russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh','kyrgyz', 
                    'nepalese', 'sri lankan', 'singaporean','bangladeshi', 'malaysian', 'fijian', 'thai', 'indonesian', 'burmese', 'vietnamese',
                    'turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian','maltese',
                    'egyptian', 'moroccan', 'tunisian','libyan', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan',
                    'zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan', 'namibian','citizen of seychelles',
                    'canadian', 'new zealander','mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander', 'costa rican','grenadian','panamanian', 'cuban',
                    'brazilian', 'colombian', 'argentinian', 'peruvian', 'venezuelan', 'ecuadorean','chilean', 'guyanese','bolivian','uruguayan']}"""

def handle_clusters(nationality, dict_clusters):
    for key in dict_clusters:
        if nationality in dict_clusters[key]:
            return key
    return 'other'

def max_per_cluster(cluster_dict, amount_names_country):  
    max_per_cluster = {}
    for key in cluster_dict:

        smallest = 1000000000000
        for country in cluster_dict[key]:
            """try:
                print(country, amount_names_country[country])
            except:
                pass"""
            if country in amount_names_country:
                if amount_names_country[country] <= smallest:
                    smallest = amount_names_country[country]
        #print(smallest)
        for country in cluster_dict[key]:
            max_per_cluster[country]=smallest
            #print(country)
            #print(smallest)
    #print(max_per_cluster)
    return max_per_cluster

if level_medium_clusters == True:
    max_per_cluster = max_per_cluster(medium_clusters_dict, amount_names_chountry)
elif level_big_clusters == True:
    max_per_cluster = max_per_cluster(big_clusters_dict, amount_names_chountry)
elif experimental == True:
    max_per_cluster = max_per_cluster(experimental_dict, amount_names_chountry)

#matrix_name_dict is a dictionary with the keys representing countries and the values being lists that store matrixes that represent names of nationals of that country.
#The keys are integers, in the dictionary nationality_to_number_dict these integers are keys to the actual country names.
matrix_name_dict = {}
nationality_to_number_dict = {}
number = 0

#this filepath must contain a file with 3 values seperates by commas in each line, the first being a nationality, the second a first name of someone with that nationality
#and the third their last name.

for key in dict_chosen_names:

    try:
        if not level_nationality == True:
            max_nat = max_per_cluster[key]
            counter = 0

        list_of_names = dict_chosen_names[key]
        random.shuffle(list_of_names)

        all_names = []
        doubles = 0

        for name in tqdm(list_of_names):
            name = name.lower()
            # remove "dr", "ms", "mr", "mrs"
            if name.split(" ")[0] == "dr" or name.split(" ")[0] == "mr" or name.split(" ")[0] == "ms" or name.split(" ")[0] == "miss" or name.split(" ")[0] == "mrs":
                space_idx = name.strip().index(" ")
                name = name[space_idx:]

            # remove weird space before name
            if list(name)[0] == " ":
                name = name[1:]

            if not level_nationality == True:
                if counter <= max_nat:
                    name = name.strip()
                    nationality = key

                    org_nat = nationality
                    if level_big_clusters == True:
                        nationality = handle_clusters(nationality, big_clusters_dict)
                    elif level_medium_clusters == True:
                        nationality = handle_clusters(nationality, medium_clusters_dict)
                    elif experimental == True:
                        nationality = handle_clusters(nationality, experimental_dict)
                    if nationality not in nationality_to_number_dict and nationality != 'other':
                        nationality_to_number_dict[nationality]= number
                        number += 1
                        matrix_name_dict[nationality_to_number_dict[nationality]]=[get_matrix_from_name(name, abc_dict)]
                    elif nationality in nationality_to_number_dict and nationality != 'other':
                        matrix_name_dict[nationality_to_number_dict[nationality]]+=[get_matrix_from_name(name, abc_dict)]       
                    counter += 1
            else:
                name = name.strip()
                nationality = key
                org_nat = nationality
                if level_big_clusters == True:
                    nationality = handle_clusters(nationality, big_clusters_dict)
                elif level_medium_clusters == True:
                    nationality = handle_clusters(nationality, medium_clusters_dict)
                elif experimental == True:
                    nationality = handle_clusters(nationality, experimental_dict)
                if nationality not in nationality_to_number_dict and nationality!= 'other':
                    nationality_to_number_dict[nationality]= number
                    number += 1
                    matrix_name_dict[nationality_to_number_dict[nationality]]=[get_matrix_from_name(name, abc_dict)]
                elif nationality in nationality_to_number_dict and nationality!= 'other':
                        matrix_name_dict[nationality_to_number_dict[nationality]]+=[get_matrix_from_name(name, abc_dict)]
        #print(doubles, "\n____")
    except:
            pass
#matrix_name_list is a list with sub-lists, each sublist containing a matrix representing a name on index 1 and a number representing a nationality on index 0.
matrix_name_list = []
nr_of_countries = 0
list_countries_used = []

# print(matrix_name_dict.keys())

minimum_per_country = min([len(matrix_name_dict[country]) for country in matrix_name_dict])
#print(minimum_per_country)
for country in matrix_name_dict:

    if len(matrix_name_dict[country]) >= minimum_per_country:

        #print(len(matrix_name_dict[country]))
        list_countries_used += [country]
        nr_of_countries += 1
        names = matrix_name_dict[country]
        random.shuffle(names)
        names = names[:minimum_per_country]
        for name in names:
            matrix_name_list += [[nr_of_countries,name]]
random.shuffle(matrix_name_list)
#print(len(matrix_name_list))
#dumping matrix_name_list as pickle file:
with open("datasets/preprocessed_datasets/final_127_matrix_name_list.pickle", "wb+") as o:
    pickle.dump(matrix_name_list, o, pickle.HIGHEST_PROTOCOL)

names_countries_used = {}
for i, element in enumerate(list_countries_used):
    #for key in nationality_to_number_dict:
        #if element == nationality_to_number_dict[key]:
            #names_countries_used += [key + " " + str(nationality_to_number_dict[key])]
    country_name = list(nationality_to_number_dict.keys())[list(nationality_to_number_dict.values()).index(element)]
    names_countries_used[country_name] = i


#print(names_countries_used)

filepath = "datasets/preprocessed_datasets/final_127_nationality_to_number_dict.json"
with open(filepath, 'w+') as f:
    json.dump(names_countries_used, f, indent=4)






#code to extract matrix name list from file again:
"""with open("datasets/preprocessed_datasets/final_prename_matrix_name_list.pickle",'rb') as o:
    matrix_name_list = pickle.load(o)

#example of how to access country-code and name:
#country code:
print(matrix_name_list[5][0])
#name:
print(get_name_from_matrix(matrix_name_list[5][1], abc_list))"""


