import numpy as np
import torch

def dice_coef_bool(gt : torch.Tensor, preds : torch.Tensor):
    """Compute the dice coefficient between two boolean arrays
    Args:
        gt (torch.Tensor): The ground-trth arrays (a batch of array)
        preds (torch.Tensor): The prediction arrays (a batch of array)

    Returns:
        torch.Tensor: tensor array of dice score for each array in the batch
    """
    # get dims
    dims = tuple(np.arange(gt.ndim-1)+1)

    # compute intersection score
    inter = torch.sum(gt * preds, dim=dims).numpy()

    # compute union
    union = (torch.sum(gt, dim=dims) + torch.sum(preds, dim=dims)).numpy()

    # compute the dice score
    score = np.zeros_like(inter, dtype="float32")
    score[union>0] = 2*inter[union>0]/union[union>0]
    score[union==0] = np.nan

    return score

def dice_coef_multiclass(gt : torch.Tensor, preds: torch.Tensor, classes):
    """In multiclass setting, computes the dice coefficient for each individual class

    Args:
        gt (torch.Tensor): The ground truth tensor (a batch of array)
        preds (torch.Tensor): The prediction tensor (a batch of array)
        classes: The list of classes to evaluate

    Returns:
        np.ndarray: an array with the dice coefficient for each class and each array in the batch
    """
    
    coefs = []
    for i in classes:
        coefs.append(dice_coef_bool(gt==i, preds==i))

    return np.array(coefs).T

def segmentation_score_stats(scores: dict):
    """Given a per batch and class score array, computes and logs sample wise and class wise statistics
    including mean and standard deviation.

    Args:
        scores dict: A dictionnary of tensor of scores
    """

    for k, v in scores.items():
        print("Mean {:s}: {:.3f} % | Std {:s}: {:.3f} %".format(k, np.nanmean(v)*100, k, np.nanstd(v)*100))
