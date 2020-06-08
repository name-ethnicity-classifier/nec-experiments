import pickle
import random

#the higher the variable number_of_names_per_country, the less nationalities are represented in the output matrix and the more names are given per nationality

#recommended number_of_names_per_country:
#level_medium_clusters: --> 5.500
#level_big_clusters: --> 42.000
number_of_names_per_country = 200

#only one of these boolean values should be true, it decides if the nationalities should be clusterred and in which way they should be clustered
level_nationality = True
level_medium_clusters = False
level_big_clusters = False

with open("dict_chosen_names.pickle",'rb') as o:
    dict_chosen_names = pickle.load(o)

#abc_dict is a dictionary where the letters "a"-"z" and " " and "-" are keys to lists representing these values in the matrix_name_list
abc_dict = {}
abc_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ","-"]
a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(abc_list)):
    abc_dict[abc_list[i]]= a[-i:]+a[:-i]


def get_matrix_from_name(name,abc_dict):
    matrix = []
    for letter in name:
        matrix += [abc_dict[letter]]
    return matrix

def get_name_from_matrix (matrix,abc_list):
    name = ""
    for letter in matrix:
        index = letter.index(1)
        letter = abc_list[index]
        name += letter
    return name

big_clusters_dict = {'european':['british', 'norwegian', 'irish', 'german', 'belgian', 'lithuanian', 'french', 'swedish', 'finnish', 'dutch', 'swiss', 'danish', 'austrian', 'estonian', 'latvian', 'luxembourger','icelandic',
                    'hungarian', 'polish', 'bulgarian', 'romanian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian',
                    'spanish', 'portugese', 'italian', 'cypriot', 'greek', 'maltese'],

                    'asian':['russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh',
                    'chinese', 'japanese', 'south korean', 'taiwanese', 'hong konger',
                    'indian', 'pakistani', 'nepalese', 'sri lankan', 'singaporean',
                    'bangladeshi', 'malaysian', 'new zealander','australian' ,'fijian', 'thai', 'filipino', 'indonesian', 'burmese', 'vietnamese'],
                    
                    'african':['turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian',
                    'egyptian', 'moroccan', 'tunisian','libyan',
                    'nigerian', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan',
                    'zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan'],  
                    
                    'american':['canadian', 'american',
                    'mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander',
                    'brazilian', 'colombian', 'argentine', 'peruvian', 'venezuelan', 'ecuadorean','chilean']}
medium_clusters_dict = {'northern_european':['british', 'norwegian', 'irish', 'german', 'belgian', 'lithuanian', 'french', 'swedish', 'finnish', 'dutch', 'swiss', 'danish', 'austrian', 'estonian', 'latvian', 'luxembourger','icelandic'],

                        'central_european':['hungarian', 'polish', 'bulgarian', 'romanian', 'czech', 'albanian', 'slovak', 'slovenian', 'algerian', 'croatian', 'serbian', 'macedonian', 'georgian', 'citizen of bosnia and herzegovina', 'kosovan', 'belarusian'],
                        
                        'southern_european':['spanish', 'portugese', 'italian', 'cypriot', 'greek', 'maltese'],
                        
                        'north_asian':['russian', 'ukrainian', 'uzbek', 'moldovan', 'turkmen','kazakh'],
                        
                        'central_asian':['chinese', 'japanese', 'south korean', 'taiwanese', 'hong konger'],
                        
                        'south_asian':['indian', 'pakistani', 'nepalese', 'sri lankan', 'singaporean'],
                        
                        'east_asian':['bangladeshi', 'malaysian', 'new zealander','australian' ,'fijian', 'thai', 'filipino', 'indonesian', 'burmese', 'vietnamese'],
                        
                        'middle_eastern':['turkish', 'iraqi', 'iranian', 'israeli', 'yemeni', 'syrian', 'afghan', 'palestinian', 'kuwaiti', 'armenian', 'bahraini', 'lebanese', 'saudi arabian', 'azerbaijani', 'emirati','omani','qatari','jordanian'],
                        
                        'north_african':['egyptian', 'moroccan', 'tunisian','libyan'],
                        
                        'central_african':['nigerian', 'cameroonian', 'ghanian', 'ugandan', 'nigerien', 'kenyan', 'gambian', 'ivorian', 'senegalese', 'eritrean', 'sierra leonean', 'congolese', 'somali', 'sudanese', 'ethiopian','angolan'],
                        
                        'south_african':['zimbabwean', 'south african', 'zambian', 'mauritian', 'malawian', 'tanzanian', 'botswanan'],
                        
                        'north_american':['canadian', 'american'],
                        
                        'central_american':['mexican', 'dominican', 'trinidadian', 'barbadian', 'kittitian', 'st lucian','jamaican','british virgin islander'],
                        
                        'south_american':['brazilian', 'colombian', 'argentine', 'peruvian', 'venezuelan', 'ecuadorean','chilean']}

def handle_clusters(nationality, dict_clusters):
    for key in dict_clusters:
        if nationality in dict_clusters[key]:
            return key
    return 'other'
            
#matrix_name_dict is a dictionary with the keys representing countries and the values being lists that store matrixes that represent names of nationals of that country.
#The keys are integers, in the dictionary nationality_to_number_dict these integers are keys to the actual country names.
matrix_name_dict = {}
nationality_to_number_dict = {}
number = 0

#this filepath must contain a file with 3 values seperates by commas in each line, the first being a nationality, the second a first name of someone with that nationality
#and the third their last name.

for key in dict_chosen_names:
    #print(key)
    #print(len(dict_chosen_names[key]))
    #print("\n")
    for name in dict_chosen_names[key]:
        name = name.strip()
        nationality = key
        org_nat = nationality
        if level_big_clusters == True:
            nationality = handle_clusters(nationality, big_clusters_dict)
        elif level_medium_clusters == True:
            nationality = handle_clusters(nationality, medium_clusters_dict)
        if nationality not in nationality_to_number_dict:
            nationality_to_number_dict[nationality]= number
            number += 1
            matrix_name_dict[nationality_to_number_dict[nationality]]=[get_matrix_from_name(name, abc_dict)]
        else:
            matrix_name_dict[nationality_to_number_dict[nationality]]+=[get_matrix_from_name(name, abc_dict)]       
#matrix_name_list is a list with sub-lists, each sublist containing a matrix representing a name on index 1 and a number representing a nationality on index 0.
matrix_name_list = []
nr_of_countries = 0
list_countries_used = []
for country in matrix_name_dict:
    if len(matrix_name_dict[country])>= number_of_names_per_country:
        list_countries_used += [country]
        nr_of_countries += 1
        names = matrix_name_dict[country]
        random.shuffle(names)
        names = names[:number_of_names_per_country]
        for name in names:
            matrix_name_list += [[country,name]]
random.shuffle(matrix_name_list)

#dumping matrix_name_list as pickle file:
with open("matrix_name_list.pickle",'wb') as o:
    pickle.dump(matrix_name_list,o,pickle.HIGHEST_PROTOCOL)

names_countries_used = []
for element in list_countries_used:
    for key in nationality_to_number_dict:
        if element == nationality_to_number_dict[key]:
            names_countries_used += [key + " " + str(nationality_to_number_dict[key])]
print(names_countries_used)

#code to extract matrix name list from file again:
#with open("matrix_name_list.pickle",'rb') as o:
#    dmatrix_name_list = pickle.load(o)

#example of how to access country-code and name:
#country code:
print(matrix_name_list[5][0])
#name:
print(get_name_from_matrix(matrix_name_list[5][1], abc_list))