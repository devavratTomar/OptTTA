import itertools
import torch
import os
import argparse
from networks.unet import UNet
from data import GenericDataset
from util.util import natural_sort
import numpy as np
from randaug.augmentations import *

STYLE_AUGMENTORS = [Gamma.__name__, GaussianBlur.__name__, Contrast.__name__, Brightness.__name__, Identity.__name__]
SPATIAL_AUGMENTORS = [RandomResizeCrop.__name__, RandomHorizontalFlip.__name__, RandomVerticalFlip.__name__, RandomRotate.__name__]

STRING_TO_CLASS = {
    'Identity': Identity,
    'GaussianBlur': GaussianBlur,
    'Contrast': Contrast,
    'Brightness': Brightness,
    'Gamma': Gamma,
    'RandomResizeCrop': RandomResizeCrop,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomRotate': RandomRotate
}

minmax_values = {
    'Identity': [0, 1],
    'GaussianBlur': [1, 5],
    'Contrast': [0.1, 1.9],
    'Brightness': [0.1, 1.9],
    'Gamma': [0.1, 2],
    'RandomResizeCrop': [0.5, 1.0],
    'RandomHorizontalFlip': [0, 1],
    'RandomVerticalFlip': [0, 1],
    'RandomRotate': [0, 1]
}

M = [2, 6, 10, 14]

@torch.no_grad()
def get_predictions(opt):
    ############################################## Output directory ############################################################
    if not os.path.exists(opt.outdir):
        os.makedirs(os.path.join(opt.outdir, 'predictions'))

    outdir = os.path.join(opt.outdir, 'predictions')
    ############################################## Load Model ##################################################################
    
    model = UNet(opt.n_channels, opt.n_classes)
    latest = natural_sort([f for f in os.listdir(os.path.join(opt.checkpoints_dir, 'saved_models')) if f.endswith('.pth')])[-1]
    print('Loading checkpoint: ', latest)
    model.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, 'saved_models', latest), map_location='cpu'))
    
    if opt.bn:
        model.train()
    else:
        model.eval()

    if opt.use_gpu:
        model = model.cuda()

    ############################################## Data Loader ##################################################################
    dataloader = GenericDataset(opt.dataroot, opt.target_sites, opt.dataset_mode, phase='test')

    ############################################## RandAug #################################################################
    all_augmentations = [Gamma(), GaussianBlur(), Contrast(), Brightness(), Identity(), RandomResizeCrop(), RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotate()]
    all_combinations = list(itertools.combinations(all_augmentations, 2))

    for i, data in enumerate(dataloader):
        img = data[0]
        pred_name = dataloader.all_segs[i]

        img = img.unsqueeze(0) # batch dimension

        all_preds = []
        
        # generate different augmentations based on m
        for m in M:
            for augs in all_combinations:
                spatial_affines = []
                style_augmentations = [f for f in augs if type(f).__name__ in STYLE_AUGMENTORS]
                spatial_augmentations = [f for f in augs if type(f).__name__ in SPATIAL_AUGMENTORS]

                # apply individual augmentations in augs
                tmp_img = img.clone()

                for aug in style_augmentations:
                    minv, maxv = minmax_values[type(aug).__name__]
                    v = torch.tensor((float(m)/30) * float(maxv - minv) + minv, dtype=torch.float32)
                    tmp_img = aug(tmp_img, v)

                for aug in spatial_augmentations:
                    minv, maxv = minmax_values[type(aug).__name__]
                    v = torch.tensor((float(m)/30) * float(maxv - minv) + minv, dtype=torch.float32)
                    tmp_img, affine = aug.test(tmp_img, v)
                    spatial_affines.append(affine)

                ## get predictions for tmp_img
                pred = model(tmp_img.cuda()).detach().cpu()

                ## inverse affine for spatial distortion
                for aug, affine in zip(reversed(spatial_augmentations), reversed(spatial_affines)):
                    pred, _ = aug.test(pred, 0, affine)

                all_preds.append(pred)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_preds = torch.softmax(all_preds, dim=1)
        all_preds = torch.mean(all_preds, dim=0)

        final_pred = torch.argmax(all_preds, dim=0)
        np.save(os.path.join(outdir, pred_name), final_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Get predictions from a trained Unet")
    parser.add_argument('--dataroot', type=str, help="Root directory of the dataset")
    parser.add_argument('--dataset_mode', type=str, choices=['spinalcord', 'heart', 'prostate'], help="type of dataset")
    parser.add_argument('--target_sites', type=str, help="Target sites to test on.")
    parser.add_argument('--use_gpu', type=str, default='true', choices=['true', 'false'])
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, help="number of classes.")
    parser.add_argument('--checkpoints_dir', type=str, help="path to the saved model")
    parser.add_argument('--outdir', type=str, help="Output directory to save the predictions")
    parser.add_argument('--bn', action='store_true')

    args = parser.parse_args()
    if args.dataset_mode == 'prostate':
        args.target_sites = ['site-' + site_nbr for site_nbr in args.target_sites.split(',')]
    else:
        args.target_sites = ['site' + site_nbr for site_nbr in args.target_sites.split(',')]
    args.use_gpu = True if args.use_gpu == 'true' else False
    
    get_predictions(args)