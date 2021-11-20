
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import torch.nn as nn
import os
from model import ConvLSTMEmbedder
from utils import device, create_dataloader, load_json, write_json


def get_embeddings(dataset_path: str, model_configuration_name: str):
    # load model configuration
    model_configuration = f"../../model_configurations/{model_configuration_name}/"
    model_file = model_configuration + "model.pt"
    classes = load_json(model_configuration + "nationalities.json")
    amount_classes = len(classes)
    hyperparameters = load_json(model_configuration + "config.json")

    # load trained model
    model = ConvLSTMEmbedder(
        class_amount=amount_classes, 
        hidden_size=hyperparameters["hidden-size"], 
        layers=hyperparameters["rnn-layers"], 
        channels=[hyperparameters["cnn-parameters"][1][0], hyperparameters["cnn-parameters"][1][0]],
        dropout_chance=0.0, 
        embedding_size=hyperparameters["embedding-size"]
    ).to(device=device)

    model.load_state_dict(torch.load(model_file))

    # create dataloader
    dataset = create_dataloader(
        dataset_path=dataset_path,
        class_amount=amount_classes
    )
    
    # iterate through all samples and save the embeddins
    all_embeddings = []
    for name, target, gender, year_of_birth in tqdm(dataset, desc="gathering embeddings", ncols=150):
        name, target = name.to(device=device), target.to(device=device)

        embedding = model.eval()(name, return_embeddings=True)
        embedding = embedding[0].cpu().detach().numpy()
        target = target[0].cpu().detach().numpy()

        all_embeddings.append([embedding, target, gender, year_of_birth])
    
    # create folder to save embeddings and classes
    embedding_folder = f"./embeddings/{model_configuration_name}/"
    if not os.path.exists(embedding_folder):
        os.mkdir(embedding_folder)
    
    # save embeddings
    open(embedding_folder + "embeddings.npy", "w+").close()
    np.save(embedding_folder + "embeddings.npy", all_embeddings, allow_pickle=True)

    # save classes
    write_json(embedding_folder + "nationalities.json", classes)


if __name__ == "__main__":
    model_configuration = "8_groups"
    get_embeddings(f"../datasets/experiment_datasets/{model_configuration}/dataset.pickle", model_configuration)
