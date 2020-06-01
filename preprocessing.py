import pickle
import random

with open("dict_chosen_names.pickle",'rb') as o:
    dict_chosen_names = pickle.load(o)

#abc_dict is a dictionary where the letters "a"-"z" and " " and "," and "-" are keys to lists representing these values in the matrix_name_list
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

#matrix_name_dict is a dictionary with the keys representing countries and the values being lists that store matrixes that represent names of nationals of that country.
#The keys are integers, in the dictionary nationality_to_number_dict these integers are keys to the actual country names.
matrix_name_dict = {}
nationality_to_number_dict = {}
number = 0

#this filepath must contain a file with 3 values seperates by commas in each line, the first being a nationality, the second a first name of someone with that nationality
#and the third their last name.

for key in dict_chosen_names:
    for name in dict_chosen_names[key]:
        name = name.strip()
        nationality = key
        if nationality not in nationality_to_number_dict:
            nationality_to_number_dict[nationality]= number
            number += 1
            matrix_name_dict[number]=[get_matrix_from_name(name, abc_dict)]
        else:
            matrix_name_dict[number]+=[get_matrix_from_name(name, abc_dict)]
        
#matrix_name_list is a list with sub-lists, each sublist containing a matrix representing a name on index 0 and a number representing a nationality on index 1.
matrix_name_list = []
for country in matrix_name_dict:
    for name in matrix_name_dict[country]:
        matrix_name_list += [[country,name]]
random.shuffle(matrix_name_list)

#dumping matrix_name_list as pickle file:
with open("matrix_name_list.pickle",'wb') as o:
    pickle.dump(matrix_name_list,o,pickle.HIGHEST_PROTOCOL)

#code to extract matrix name list from file again:
#with open("matrix_name_list.pickle",'rb') as o:
#    dmatrix_name_list = pickle.load(o)

#example of how to access country-code and name:
#country code:
print(matrix_name_list[10][0])
#name:
print(get_name_from_matrix(matrix_name_list[10][1], abc_list))
