
from train_setup import TrainSetup
import argparse


model_config = {
    # name of the model/experiment in general (choose descriptive name, the .pt file will have the same name and store the parameters of the model)
    "model-name": "my_model",

    # path to the dataset folder (must contain "matrix_name_list.pickle" and "nationality_classes.json" and be stored in "../datasets/preprocessed_datasets/")
    "dataset-name": "14_nat_and_else",
    
    # percentage of the test and validation set
    "test-size": 0.05,

    # name of the optimizer (changing "optimizer" in this config won't make a difference, the optimizer has to be changed in the "train_setup.py" by hand)
    "optimizer": "Adam",

    # name of the loss function (changing "loss-function" in this config won't make a difference, the loss function has to be changed in the "train_setup.py" by hand)
    "loss-function": "NLLLoss",

    # amount of epochs
    "epochs": 15,

    # batch size
    "batch-size": 1000,

    # initial learning rate (DON'T change when resuming the training, change "lr-schedule[0]" instead!)
    "init-learning-rate": 0.0035,

    # cnn parameters (idx 0: amount of layers, idx 1: kernel size, idx 2: list of feature map dimensions)
    "cnn-parameters": [1, 3, [64]],
    
    # hidden size of the LSTM
    "hidden-size": 200, 

    # amount of layers inside the LSTM
    "rnn-layers": 2,

    # learning-rate parameters (idx 0: current lr, idx 1: decay rate, idx 2: decay intervall in iterations), 
    # change current lr when resuming the training to the learning rate of the last checkpoint
    "lr-schedule": [0.0029, 0.9875, 100],

    # dropout change of the LSTM output
    "dropout-chance": 0.3,

    # embedding size ("embedding-size" x 1)
    "embedding-size": 200,

    # augmentation chance (name part switching will slow down the training process when set high)
    "augmentation": 0.5,

    # when resume is true: replace the first element of "lr-schedule" (the current lr) with the learning rate of the last checkpoint
    "resume": True
}


# for automatic training read the dataset and model configuration as flags
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasetPath", required=False, help="path to the dataset folder")
parser.add_argument("-m", "--modelPath", required=False, help="path to the model file")
parser.add_argument("-n", "--sessionName", required=False, help="name for this model")

args = vars(parser.parse_args())
if args["datasetPath"] != None:
    model_config["dataset-name"] = args["datasetPath"]
if args["modelPath"] != None:
    model_config["model-file"] = args["modelPath"]
if args["sessionName"] != None:
    model_config["model-name"] = args["sessionName"]

# train and test
train_setup = TrainSetup(model_config)
train_setup.train()
train_setup.test(print_amount=500, plot_confusion_matrix=True, plot_scores=True)
