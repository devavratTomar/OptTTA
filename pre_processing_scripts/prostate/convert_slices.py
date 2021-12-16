import nibabel as nib
import numpy as np
from PIL import Image
import argparse

import os

def normalize_img(vol):
    vol = (vol - vol.min())/(vol.max() - vol.min())
    return vol

def convert_slices(rootdir, outdir, debug=False):
    all_sites = [f for f in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, f))]
    print(all_sites)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if debug and not os.path.exists(os.path.join(outdir, 'debug')):
        os.makedirs(os.path.join(outdir, 'debug'))

    for site in all_sites:
        imgs = sorted([p for p in os.listdir(os.path.join(rootdir, site)) if p.endswith('nii.gz') and 'segmentation' not in p.lower()])
        segs = sorted([p for p in os.listdir(os.path.join(rootdir, site)) if p.endswith('nii.gz') and 'segmentation' in p.lower()])
        for imgp, segp in zip(imgs, segs):
            imgvol = nib.load(os.path.join(rootdir, site, imgp)).get_fdata()
            imgvol = normalize_img(imgvol)
            
            segvol = nib.load(os.path.join(rootdir, site, segp)).get_fdata()
            print(site, np.unique(segvol))

            # tmpsegvol = np.zeros_like(segvol)
            # tmpsegvol[segvol !=0] = 1
            # segvol = tmpsegvol

            # assert segvol.shape == imgvol.shape
            # print(segvol.shape, np.unique(segvol))


            # for j in range(imgvol.shape[2]):
            #     imgslice = imgvol[:, :, j]
            #     segslice = segvol[:, :, j]

            #     np.save(os.path.join(outdir, 'site-' + site + '-' + imgp.lower() + str(j)), imgslice)
            #     np.save(os.path.join(outdir, 'site-' + site + '-' + segp.lower() + str(j)), segslice)

            #     if debug:
            #         imgdebug = (imgslice * 255).astype(np.uint8)
            #         segdebug = (segslice * 50).astype(np.uint8)

            #         Image.fromarray(imgdebug).save(os.path.join(outdir, 'debug', 'site-' + site + '-' + imgp.lower() + str(j) + '.png'))
            #         Image.fromarray(segdebug).save(os.path.join(outdir, 'debug', 'site-' + site + '-' + segp.lower() + str(j) + '.png'))

parser = argparse.ArgumentParser('Preprocess Prostate')
parser.add_argument('--rootdir', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

convert_slices(args.rootdir, args.outdir, args.debug)