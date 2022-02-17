import argparse
import os
import albumentations as A

import torch
import numpy as np
from util.util import natural_sort
import re
from scipy import ndimage

from util import dice_coef_multiclass, segmentation_score_stats

def get_largest_component(image):
    """
    get the largest component from 2D or 3D binary image
    image: nd array
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return  output 

def is_site(sites, name):
    for site in sites:
        if site in name:
            return True
    
    return False

@torch.no_grad()
def compute_metrics_one_step(pred, seg, all_classes):

    metrics_dic = {}

    # compute dice coefficients
    batch_dice_coef = dice_coef_multiclass(seg, pred, all_classes)

    # compute class-wise
    for i, coef in enumerate(batch_dice_coef.T):
        metrics_dic["ds_class_{:d}".format(i)] = torch.tensor(coef)

    # compute sample-wise mean (w/o background)
    metrics_dic["ds"] = torch.tensor(np.nanmean(batch_dice_coef[:,1:], axis=1))

    return metrics_dic


parser = argparse.ArgumentParser(description="Test script")

# Checkpoint arguments
parser.add_argument('--prediction_path', type=str)

# Model arguments
parser.add_argument('--n_classes', default=3, type=int)

# Dataset arguments
parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate'], type=str)
parser.add_argument('--dataroot', type=str)
parser.add_argument('--target_sites', default='4')

opt = parser.parse_args()

if opt.dataset_mode == 'prostate':
    opt.target_sites = ['site-'+ site_nbr for site_nbr in opt.target_sites.split(',')]
else:
    opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]

# Extract predictions and gt segments
predictions_path = os.path.join(opt.prediction_path, "predictions")
gts_path = os.path.join(opt.dataroot)

predictions_flist = np.array(natural_sort([f for f in os.listdir(predictions_path) if is_site(opt.target_sites, f)]))

if opt.dataset_mode == 'spinalcord':
    patient_roots = np.unique(np.array(natural_sort(['-'.join(x.split('-')[:2]) for x in predictions_flist])))
elif opt.dataset_mode == 'heart':
    patient_roots = np.unique(np.array(natural_sort([x.split('_')[0] for x in predictions_flist])))
elif opt.dataset_mode == 'prostate':
    patient_roots = np.unique(np.array(natural_sort([x.split(".")[0] for x in predictions_flist])))
else:
    raise Exception('Unrecognized dataset mode.')


print(patient_roots)

# Compute scores for the predictions
metrics = None

for patient in patient_roots:

    patient_f_slices = np.array(natural_sort([f for f in predictions_flist if patient in f]))
    all_preds = torch.Tensor()
    all_gts = torch.Tensor()

    for slice in patient_f_slices:
        pred = np.load(os.path.join(predictions_path, slice)).astype(np.float64)
        gt = np.load(os.path.join(opt.dataroot, slice))

        # Resize gt
        gt = A.Resize(256, 256)(image=gt, mask=gt)["mask"]

        # Make tensors
        gt = torch.from_numpy(gt).to(torch.long).unsqueeze(0)
        pred = torch.from_numpy(pred).to(torch.long).unsqueeze(0)

        # Aggregate predictions and gts
        all_preds = torch.cat([all_preds, pred])
        all_gts = torch.cat([all_gts, gt])
        
    all_classes = np.arange(opt.n_classes)

    all_gts = all_gts.unsqueeze(0)
    all_preds = all_preds.unsqueeze(0)

    # Compute metrics on the 3D volume
    if metrics is None:
        metrics = compute_metrics_one_step(all_preds, all_gts, all_classes)
    else:
        for k, v in compute_metrics_one_step(all_preds, all_gts, all_classes).items():
            metrics[k] = torch.cat((metrics[k], v))

segmentation_score_stats(metrics)
