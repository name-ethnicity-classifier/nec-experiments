import json

#abc_dict is a dictionary where the letters "a"-"z" and " " and "," and "-" are keys to lists representing these values in the matrix_name_list
abc_dict = {}
abc_list = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"," ",",","-", "'",'’', 'é', 'è','ñ','í','ó','á','`','ü','ö','ã','ä','ø','å','õ','æ','ń','ć','ò','ë','ç','ş','ğ','@','š','č','ž','ð','þ','ú','ě','ý','ň','ř']
a = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(abc_list)):
    abc_dict[abc_list[i]]= a[-i:]+a[:-i]


def get_matrix_from_name(name,abc_dict):
    matrix = []
    for letter in name:
        if letter not in abc_list:
            letter = " "
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
filepath = "PATHNAME"
with open (filepath, "r") as f:
    for line in f.readlines():
        line = line.strip().split(",")
        nationality = line[0]
        first_name = line[1]
        last_name = line[2]
        if nationality not in nationality_to_number_dict:
            nationality_to_number_dict[nationality]= number
            number += 1
            matrix_name_dict[number]=[get_matrix_from_name((first_name + "," +last_name), abc_dict)]
        else:
            matrix_name_dict[number]+=[get_matrix_from_name((first_name + "," +last_name), abc_dict)]
        
#matrix_name_list is a list with sub-lists containing information on one nationality, 
#each of these sublists containing the number representing the nationality on index 0
#and names of people with this nationality stored in a list at index 1, each name being a matrix.
matrix_name_list = []
for country in matrix_name_dict:
    matrix_name_list += [[country, matrix_name_dict[country]]]

#examples of how to access country-code and name:
print("A name from country with code ", matrix_name_list[0][0], " is ",get_name_from_matrix(matrix_name_list[0][1][0], abc_list))
print("A name from country with code ", matrix_name_list[3][0], " is ",get_name_from_matrix(matrix_name_list[3][1][0], abc_list))
print("A name from country with code ", matrix_name_list[20][0], " is ",get_name_from_matrix(matrix_name_list[20][1][52], abc_list))

json.dumps(matrix_name_list)