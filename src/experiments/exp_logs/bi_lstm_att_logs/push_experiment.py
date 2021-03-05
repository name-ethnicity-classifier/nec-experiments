import wandb
import json



def extract_exp_information(train_stats_file: str) -> list:
    with open(train_stats_file, "r") as f:
        train_stats = json.load(f)
    
    # load all hyperparameters
    hyperparameters = train_stats["hyperparameters"]

    # handle learning-rate logs
    lr_steps = [lst[0] for lst in hyperparameters["learning-rate"][1:]]
    hyperparameters["init-learning-rate"] = hyperparameters["learning-rate"][0]

    # load validation stats
    val_acc = train_stats["results"]["validation-accuracy"]
    val_loss = train_stats["results"]["validation-loss"]

    # load train stats
    train_acc = train_stats["results"]["train-accuracy"]
    train_loss = train_stats["results"]["train-loss"]

    return hyperparameters, val_acc, val_loss, train_acc, train_loss, lr_steps


def post_run_push(train_stats_file: str) -> None:
    hyperparameters, val_acc, val_loss, train_acc, train_loss, lr_steps = extract_exp_information(train_stats_file)
    epochs = len(val_acc)
    
    wandb.init(project="name-ethnicity-classification", entity="theodorp", config=hyperparameters)

    """for lr_step in lr_steps:
        wandb.log({"learning-rate": lr_step})
        print(f"lr-step: {lr_step}")"""
    
    print("\n---------------------\n")

    for step in range(epochs):
        wandb.log({"validation-accuracy": val_acc[step], "validation-loss": val_loss[step], "train-accuracy": train_acc[step], "train-loss": train_loss[step]})
        print(f"val_acc: {val_acc[step]}")
        print(f"val_loss: {val_loss[step]}")
        print(f"train_acc: {train_acc[step]}")
        print(f"train_loss: {train_loss[step]}")

        print("\n---------------------\n")


exp_log = "experiment2_bi_lstm_attention_concat/train_stats.json"
post_run_push(exp_log)