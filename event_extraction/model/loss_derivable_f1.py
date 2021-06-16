"""Function for derivable F1 score loss."""

import torch

def loss_macro_f1(y_true, y_pred, mask):
    """Compute derivated macro F1 loss.

    Args:
        y_true (torch.tensor): true labels with the label name for each input.
        y_pred (torch.tensor): prediction vector with percentage for each class, already softmaxed.
        mask (torch.tensor): mask to remove unwanted tokens.

    Returns:
        torch.tensor: the value of the loss.
    """
    y_true = y_true * mask.to(torch.int64)

    sum_f1 = 0
    for class_id in range(len(y_pred[0])):
        class_f1 = masked_binary_macro_f1(
            y_pred[range(y_pred.shape[0]), class_id], y_true == class_id, mask
        )
        sum_f1 += class_f1

    loss = sum_f1 / len(y_pred[0])

    return loss

def masked_binary_macro_f1(y_true, y_pred, mask):
    """Idea from Benoit Favre + https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric.

    Compute the F1 of a specific class.

    Args:
        y_true (torch.tensor): true labels with the label name for each input.
        y_pred (torch.tensor): prediction vector with percentage for each class, already softmaxed.
        mask (torch.tensor): mask to remove unwanted tokens.

    Returns:
        torch.tensor: the value of the loss.
    """
    epsilon = 1e-7
    
    tp, tn, fp, fn = compute_tp_fn_fp_fn(y_true, y_pred, mask)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    loss = 1 - f1
    return loss

def loss_micro_f1(y_true, y_pred, mask):
    """Compute derivable micro-f1 loss in pytorch.

    Args:
        y_true (torch.tensor): true labels with the label name for each input.
        y_pred (torch.tensor): prediction vector with percentage for each class, already softmaxed.
        mask (torch.tensor): mask to remove unwanted tokens.

    Returns:
        torch.tensor: the value of the loss.
    """
    y_true = y_true * mask.to(torch.int64)
    loss = masked_binary_micro_f1(y_true, y_pred, mask)
    return loss

def masked_binary_micro_f1(y_true, y_pred, mask):
    """Idea from Benoit Favre + https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric.

    Compute the micro-F1.

    Args:
        y_true (torch.tensor): true labels with the label name for each input.
        y_pred (torch.tensor): prediction vector with percentage for each class, already softmaxed.
        mask (torch.tensor): mask to remove unwanted tokens.

    Returns:
        torch.tensor: the value of the loss.
    """
    epsilon = 1e-7
    
    sum_tp = 0
    sum_tn = 0
    sum_fp = 0
    sum_fn = 0

    for class_id in range(len(y_pred[0])):
        tp, tn, fp, fn = compute_tp_fn_fp_fn(
            (y_true == class_id).float(), y_pred[range(y_pred.shape[0]), class_id], mask
        )
        sum_tp += tp
        sum_tn += tn
        sum_fp += fp
        sum_fn += fn
    

    precision = sum_tp / (sum_tp + sum_fp + epsilon)
    recall = sum_tp / (sum_tp + sum_fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    loss = 1 - f1

    return loss

def compute_tp_fn_fp_fn(y_true, y_pred, mask):
    """Compute true positive, true negative, false positive and false negative.

    Args:
        y_true (torch.tensor): true labels with the label name for each input.
        y_pred (torch.tensor): prediction vector with percentage for each class, already softmaxed.
        mask (torch.tensor): mask to remove unwanted tokens.

    Returns:
        (torch.tensor, torch.tensor, torch.tensor, torch.tensor): 
            true positive, true negative, false positive and false negative
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    tp = (y_true * y_pred * mask).sum().float()
    tn = ((1 - y_true) * (1 - y_pred) * mask).sum().float()
    fp = ((1 - y_true) * y_pred * mask).sum().float()
    fn = (y_true * (1 - y_pred) * mask).sum().float()

    return tp, tn, fp, fn