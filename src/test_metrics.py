
import numpy as np
import matplotlib.pyplot as plt


def validate_accuracy(y_true, y_pred, threshold: float) -> float:
    """ calculate the accuracy of predictions
    
    :param torch.tensor y_true: targets
    :param torch.tensor y_pred: predictions
    :param float threshold: treshold for logit-rounding
    :return float: accuracy
    """

    correct_in_batch = 0
    for idx in range(len(y_true)):
        output, target = y_pred[idx], y_true[idx]

        amount_classes = output.shape[0]

        target_empty = np.zeros((amount_classes))
        target_empty[target] = 1
        target = target_empty

        output = list(output).index(max(output))
        output_empty = np.zeros((amount_classes))
        output_empty[output] = 1
        output = output_empty

        # output = list(np.exp(output))
        # output = [1 if e >= threshold else 0 for e in output]

        if list(target) == list(output):
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


def plot(train_acc: list, train_loss: list, val_acc: list, val_loss: list, save_to: str="") -> None:
    """ plots training stats
    
    :param list train_acc/train_loss: training accuracy and loss
    :param list val_acc/val_loss: validation accuracy and loss
    """

    plt.style.use("ggplot")
    fig, axs = plt.subplots(2)
    xs = range(1, (len(train_acc) + 1))

    axs[0].plot(xs, train_acc, "r", label="train-acc")
    axs[0].plot(xs, val_acc, "b", label="val-acc")
    axs[0].legend()
    axs[0].set_title("train-/ val-acc")

    axs[1].plot(xs, train_loss, "r", label="train-loss")
    axs[1].plot(xs, val_loss, "b", label="val-loss")
    axs[1].legend()
    axs[1].set_title("train-/ val-loss")
    
    if save_to != "":
        plt.savefig(save_to)

    plt.show()


def create_confusion_matrix(y_true: list, y_pred: list, classes: dict={}) -> None:
    """ creates and plots a confusion matrix given two list (targets and predictions)

    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    :param dict classes: a dictionary of the countries with they index representation
    """

    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx][0]

        output = y_pred[idx]
        output = list(output).index(max(output))

        confusion_matrix[target][output] += 1

    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(list(classes.keys()))))
    ax.set_yticks(np.arange(len(list(classes.keys()))))

    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()


        
