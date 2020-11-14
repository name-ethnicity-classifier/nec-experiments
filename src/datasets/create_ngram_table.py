
import numpy as np
import json


raw_alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
alphabet_indices = [str(i) for i in list(range(1, len(raw_alphabet) + 1))]


table = {}
idx_counter = 1
for letter in alphabet_indices:
    fst_letter = letter
    
    for sec_letter in alphabet_indices:
        table[str(fst_letter) + "$" + str(sec_letter)] = idx_counter

        idx_counter += 1

table["27"] = idx_counter + 1


with open("./bi_gram_table.json", "w") as f:
    json.dump(table, f, indent=4)

