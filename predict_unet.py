import torch
import os
import argparse
from networks.unet import UNet
from data import GenericDataset
from util.util import natural_sort
import numpy as np

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
    print(len(dataloader.all_imgs))

    for i, data in enumerate(dataloader):
        img = data[0]
        pred_name = dataloader.all_segs[i]

        if opt.use_gpu:
            img = img.cuda()

        img = img.unsqueeze(0) # batch dimension
        pred = model(img).detach()
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze(0) # remove batch dimension

        pred = pred.cpu().numpy()
        np.save(os.path.join(outdir, pred_name), pred)


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