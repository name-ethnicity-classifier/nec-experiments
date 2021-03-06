
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import numpy as np
import json
import pandas as  pd
import string

import sys
# insert at 1, 0 is for other usage
sys.path.insert(1, "src/final_model/")

from src.final_model.model import ConvLSTM as Model

# check if nvidia GPU is available, if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def console_parser() -> list:
    """ handles console arguments

    :return list: list of names to predict ethnicities
    :return str: path of csv-file in which to save ethnicities (also works as a check if a certain flag is set)
    """

    # get name from console argument parser
    parser = argparse.ArgumentParser()
    # arg. option 1, ie. : python predict_ethnicity.py -n "theodor peifer"
    parser.add_argument("-n", "--name", required=False, help="Parse name to predict its ethnicity.")
    # arg. option 2, ie. : python predict_ethnicity.py -c "names.csv"
    parser.add_argument("-c", "--csv", required=False, nargs=2, help="Parse a csv file with names to predict their enthnicities and another csv file to save them.")

    # set to 'True' if the -c argument is used
    csv_out_path = ""

    # arguments to dict
    args = vars(parser.parse_args())

    # check if -/--name is used and -c/--csv not
    if args["name"] != None and args["csv"] == None:
        names = [args["name"]]

    # check if -/--name is not used but -c/--csv is
    elif args["name"] == None and args["csv"] != None:
        csv_in_path = args["csv"][0]
        csv_out_path = args["csv"][1]
        names = pd.read_csv(csv_in_path)["names"].tolist()

    # check if -/--name and -c/--csv are both not used (raise error)
    elif args["name"] == None and args["csv"] == None:
        raise TypeError("Either -n/--name or -c/--csv must be set!")

    # check if -/--name and -c/--csv are both used (raise error)
    elif args["name"] != None and args["csv"] != None:
        raise TypeError("-n/--name and -c/--csv can't both be set!")

    return names, csv_out_path


def preprocess_names(names: list=[str]) -> torch.tensor:
    """ create a pytorch-usable input-batch from a list of string-names
    
    :param list names: list of names (strings)
    :return torch.tensor: preprocessed names (to tensors, padded, encoded)
    """

    sample_batch = []
    for name in names:

        # create index-representation from string name, ie: "joe" -> [10, 15, 5], indices go from 1 ("a") to 28 ("-")
        alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]
        int_name = []
        for char in name:
            int_name.append(alphabet.index(char.lower()) + 1)
        
        name = torch.tensor(int_name)
        sample_batch.append(name)

    padded_batch = pad_sequence(sample_batch, batch_first=True)

    padded_to = list(padded_batch.size())[1]
    padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 1).to(device=device)

    return padded_batch

def predict(input_batch, model_conifg: dict) -> str:
    """ load model and predict preprocessed name

    :param torch.tensor input_batch: input-batch
    :param str model_path: path to saved model-paramters
    :param dict classes: a dictionary containing all countries with their class-number
    :return str: predicted ethnicities
    """

    # prepare model (map model-file content from gpu to cpu if necessary)
    model = Model(
                class_amount=model_conifg["amount-classes"], 
                embedding_size=model_conifg["embedding-size"],
                hidden_size=model_conifg["hidden-size"],
                layers=model_conifg["rnn-layers"],
                kernel_size=model_conifg["cnn-parameters"][0],
                channels=model_conifg["cnn-parameters"][1]
            ).to(device=device)

    model_path = model_conifg["model-file"]

    if device != "cuda:0":
        model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
    else:
        model.load_state_dict(torch.load(model_path))

    model = model.eval()

    # classify names
    predictions = model(input_batch.float())

    # convert numerics to country name
    predicted_ethnicites = []
    for idx in range(len(predictions)):
        prediction = predictions.cpu().detach().numpy()[idx]
        prediction_idx = list(prediction).index(max(prediction))
        ethnicity = list(classes.keys())[list(classes.values()).index(prediction_idx)]
        predicted_ethnicites.append(ethnicity)

    return predicted_ethnicites
    

if __name__ == "__main__":
    # get names from console arguments
    names, csv_out_path = console_parser()

    # get dictionary of classes
    with open("src/datasets/preprocessed_datasets/final_more_nationalities/nationality_classes.json", "r") as f: classes = json.load(f)

    # preprocess inputs
    input_batch = preprocess_names(names=names)
    
    model_config = {
        "model-file": "src/final_model/models/model7.pt",
        "amount-classes": len(classes),
        "embedding-size": 200,
        "hidden-size": 200,
        "rnn-layers": 2,
        "cnn-parameters": [3, [64]],
    }

    # predict ethnicities
    ethnicities = predict(input_batch, model_config)

    # check if the -c/--csv flag was set, by checking if there is a csv-save-file, if so: save names with their ethnicities
    if len(csv_out_path) > 0:
        df = pd.DataFrame()
        df["names"] = names
        df["ethnicities"] = ethnicities

        open(csv_out_path, "w+").close()
        df.to_csv(csv_out_path, index=False)
    
    # if a single name was parsed using -n/--name, print the predicition
    else:
        print("\nname:", names[0], "- predicted ethnicity:", ethnicities[0])





    




