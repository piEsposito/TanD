import torch
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, \
    average_precision_score


def get_preds_labels_scores(dataloader,
                            classifier,
                            device):
    predictions_list = []
    labels_list = []
    scores_list = []

    with torch.no_grad():
        for data in dataloader:
            features, labels = data
            outputs = classifier(features.to(device))
            _, predicted = torch.max(outputs.data, 1)

            predictions_list.append(predicted)
            labels_list.append(labels)
            scores_list.append(outputs)

    y_true = torch.cat(labels_list).cpu().numpy()
    y_pred = torch.cat(predictions_list).cpu().numpy()
    scores = torch.cat(scores_list).cpu().numpy()

    return y_true, y_pred, scores


def eval_model_binary(y_true,
                      y_pred,
                      label_name=""):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    specificity = 1 - fpr
    npv = tn / (tn + fn)

    return {
        f"{label_name}_precision": precision,
        f"{label_name}_recall": recall,
        f"{label_name}_fpr": fpr,
        f"{label_name}_specificity": specificity,
        f"{label_name}_npv": npv
    }


def eval_model_per_class(y_true,
                         y_pred,
                         label_names):
    metrics = {}
    for i in range(len(label_names)):
        label_name = label_names[i]
        y_true_class = (y_true == i).astype(np.float32)
        y_pred_class = (y_pred == i).astype(np.float32)

        class_metrics = eval_model_binary(y_true_class,
                                          y_pred_class,
                                          label_name)
        metrics.update(class_metrics)
    return metrics


def save_confusion_matrix(y_true,
                          y_pred,
                          label_names,
                          fname="confusion_matrix.png"):
    ax = plt.subplot()
    conf_matrix = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_matrix, annot=True, ax=ax)

    plt.title("Confusion matrix for ")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    ax.xaxis.set_ticklabels(label_names)
    ax.yaxis.set_ticklabels(label_names)

    plt.savefig(fname)
    return fname


def save_roc_curve(y_true,
                   scores,
                   label_names,
                   fname="roc_curve.png"):
    lw = 2
    class_nbr = len(label_names)

    if class_nbr == 2:
        y_true_binarized = label_binarize(y_true, [0, 1, 2])[:, :2]
    else:
        y_true_binarized = label_binarize(y_true, [i for i in range(class_nbr)])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_nbr):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_nbr)]))

    # Plot all ROC curves
    plt.figure()

    for i, color in zip(range(class_nbr), label_names):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for all classes')
    plt.legend(loc="lower right")
    plt.savefig(fname)
    plt.clf()

    return fname


def save_pr_curve(y_true,
                  scores,
                  label_names,
                  fname="pr_curve.png"):

    class_nbr = len(label_names)

    if class_nbr == 2:
        y_true_binarized = label_binarize(y_true, [0, 1, 2])[:, :2]
    else:
        y_true_binarized = label_binarize(y_true, [i for i in range(class_nbr)])

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(class_nbr):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], scores[:, i])
        average_precision[i] = average_precision_score(y_true_binarized[:, i], scores[:, i])

    ########

    # Plot all ROC curves

    for i, color in zip(range(class_nbr), label_names):
        plt.plot(recall[i], precision[i], lw=2,
                 label='Precision-recall for class {0} (area = {1:0.2f})'
                       ''.format(label_names[i], average_precision[i]))

    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for all classes')

    plt.savefig(fname)
    plt.clf()

    return fname
