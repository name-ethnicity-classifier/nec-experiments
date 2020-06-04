
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import argparse
import numpy as np

from model import Model
from utils import onehot_to_string, string_to_onehot


# get name from console argument parser (ie: python deploy.py -n "franziska hafner")
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", required=True, help="Parse name to predict its ethnicity.")
name = vars(parser.parse_args())["name"]

# check if nvidia GPU is available, if not, use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_name() -> list:
    """ string type name -> tensor type one-hot enc. name """

    # convert string name to one-hot encoded name
    one_hot_name = string_to_onehot(string_name=name.lower())
    # convert to torch tensor (on gpu)
    one_hot_name = torch.tensor(one_hot_name).to(device=device)
    # create batch (size = 1)
    one_hot_name = one_hot_name.reshape(1, *one_hot_name.size())

    return one_hot_name

def predict(input_tensor, model_path: str="", classes: dict={}) -> str:
    """ load model and predict preprocessed name

    :param torch.tensor input_tensor: input
    :param str model_path: path to saved model-paramters
    :param dict classes: a dictionary containing all countries with their class-number
    :return str: predicted ethnicity
    """

    # prepare model
    model = Model().cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # predict and convert to country name
    prediction = model(input_tensor.float(), len(input_tensor[0]), 1)
    prediction = prediction.cpu().detach().numpy()[0]
    prediction_idx = list(prediction).index(max(prediction))
    ethnicity = list(classes.keys())[list(classes.values()).index(prediction_idx)]

    return ethnicity


classes = {'british': 0, 'indian': 1, 'hungarian': 2, 'irish': 3, 'canadian': 4, 'spanish': 5, 'american': 6, 'german': 7, 'zimbabwean': 8, 'portugese': 9, 'polish': 10, 'bulgarian': 11, 'bangladeshi': 12, 'malaysian': 13, 'turkish': 14, 'belgian': 15, 'pakistani': 16, 'italian': 17, 'romanian': 18, 'new zealander': 19, 'lithuanian': 20, 'french': 21, 'australian': 22, 'chinese': 23, 'swedish': 24, 'czech': 25, 'nigerian': 26, 'greek': 27, 'south african': 28, 'dutch': 29, 'sri lankan': 30, 'ukrainian': 31, 'swiss': 32, 'danish': 33, 'ghanian': 34, 'slovak': 35, 'russian': 36, 'austrian': 37, 'latvian': 38, 'brazilian': 39, 'filipino': 40, 'jamaican': 41}
preprocessed_name = preprocess_name()
ethnicity = predict(preprocessed_name, model_path="models/model1.pt", classes=classes)

print("\nname:", name, "- predicted ethnicity:", ethnicity)






    




