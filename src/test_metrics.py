
import numpy as np
import matplotlib.pyplot as plt


def validate_accuracy(y_true: list, y_pred: list, threshold: float) -> float:
    """ calculates the accuracy of predictions
    
    :param list y_true: targets
    :param list y_pred: predictions
    :param float threshold: treshold for logit-rounding
    :return float: accuracy
    """

    correct_in_batch = 0
    for idx in range(len(y_true)):
        output, target = y_pred[idx], y_true[idx]

        if target == output:
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


def create_confusion_matrix(y_true: list, y_pred: list, classes: dict={}) -> None:
    """ creates and plots a confusion matrix given two list (targets and predictions)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param dict classes: a dictionary of the countries with they index representation
    """

    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx]
        output = y_pred[idx]

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


def precision(y_true: list, y_pred: list, classes: int=10) -> list:
    """ calculates recall scores of classes (against all other classes)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param int classes: amount of classes
    :return list: list of the precision scores of each class
    """

    total_prediction_of_classes, total_true_prediction_of_classes = [0 for i in range(classes)], [0 for i in range(classes)]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]

        for class_ in range(classes):
            if output == class_:
                total_prediction_of_classes[class_] += 1

                if output == target:
                    total_true_prediction_of_classes[class_] += 1

    all_precisions = [0 for i in range(classes)]
    for i in range(classes):
        if total_prediction_of_classes[i] > 0:
            all_precisions[i] = round((total_true_prediction_of_classes[i] / total_prediction_of_classes[i]), 5)
        else:
            all_precisions[i] = 0

    return all_precisions


def recall(y_true: list, y_pred: list, classes: int=10) -> list:
    """ calculates recall scores of all classes (against all other classes)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param int classes: amount of classes
    :return list: list of the recall scores of each class
    """

    total_prediction_of_classes, total_true_of_classes = [0 for i in range(classes)], [0 for i in range(classes)]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]

        for class_ in range(classes):
            if target == class_:
                total_true_of_classes[class_] += 1

                if output == class_:
                    total_prediction_of_classes[class_] += 1

    all_recalls = [0 for i in range(classes)]
    for i in range(classes):
        if total_true_of_classes[i] > 0:
            all_recalls[i] = round((total_prediction_of_classes[i] / total_true_of_classes[i]), 5)
        else:
            all_recalls[i] = 0

    return all_recalls


def f1_score(precision_scores: list, recall_scores: list) -> list:
    """ calculates F1 scores of all classes (against all other classes)

    :param list precision_scores: list containing the precision of each class
    :param list recall_scores: list containing the recall of each class
    :return list: list of the F1 score of each class
    """
    f1_scores = []
    for i in range(len(precision_scores)):
        precision_score, recall_score = precision_scores[i], recall_scores[i]

        f1_score = 2 * ((precision_score * recall_score) / (precision_score + recall_score))
        f1_scores.append(f1_score)

    return f1_scores


def plot(train_acc: list, train_loss: list, val_acc: list, val_loss: list, save_to: str="") -> None:
    """ plots training stats NOT IN USAGE
    
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