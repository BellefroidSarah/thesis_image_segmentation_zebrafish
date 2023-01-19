import torch


# Computes the precision between the prediction (tensor) and groundtruth (tensor)
def precision(prediction, groundtruth, eps=1E-8):
    tp, fp, tn, fn = confusion(prediction, groundtruth)
    return tp / (tp + fp + eps)


# Computes the recall between the prediction (tensor) and groundtruth (tensor)
def recall(prediction, groundtruth, eps=1E-8):
    tp, fp, tn, fn = confusion(prediction, groundtruth)
    return tp / (tp + fn + eps)


# Computes the F1/Dice score between the prediction (tensor) and groundtruth (tensor)
def F1Score(prediction, groundtruth, eps=1E-8):
    tp, fp, tn, fn = confusion(prediction, groundtruth)
    return (2 * tp) / (2 * tp + fp + fn + eps)


# Computes the IOU score between the prediction (tensor) and groundtruth (tensor)
def IOUScore(prediction, groundtruth, eps=1E-8):
    tp, fp, tn, fn = confusion(prediction, groundtruth)
    return tp / (tp + fp + fn + eps)


# Computes the confusion matrix betwenn the prediction (tensor) and truth (tensor).
# Returns the number of true positives, false positives, true negatives and false negatives.
# Source:
# https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    prediction = prediction.reshape(-1)
    truth = truth.reshape(-1)

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives
