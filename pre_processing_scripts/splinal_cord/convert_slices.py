import nibabel as nib
import os
import argparse
from PIL import Image
import numpy as np

def slice_volumes(rootdir, outdir, debug=False):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        if debug:
            os.makedirs(os.path.join(outdir, 'debug'))

    all_imgs = sorted([f for f in os.listdir(rootdir) if 'image' in f])
    all_segs = sorted([f for f in os.listdir(rootdir) if 'mask' in f])

    for f_img, f_seg in zip(all_imgs, all_segs):
        img_vol = nib.load(os.path.join(rootdir, f_img)).get_fdata()
        seg_vol = nib.load(os.path.join(rootdir, f_seg)).get_fdata()

        assert img_vol.shape == seg_vol.shape

        nslices = img_vol.shape[2]

        for i in range(nslices):
            np_img_slice = img_vol[:, :, i]
            np_seg_slice = seg_vol[:, :, i]

            np.save(os.path.join(outdir, f_img[:-7] + str(i)), np_img_slice)
            np.save(os.path.join(outdir, f_seg[:-7] + str(i)), np_seg_slice)

            if debug:
                debug_img = 255*np_img_slice
                debug_slice = 100*np_seg_slice
                debug_img = debug_img.astype(np.uint8)
                debug_slice = debug_slice.astype(np.uint8)
                Image.fromarray(debug_img).save(os.path.join(outdir, 'debug', f_img[:-7] + str(i) + '.png'))
                Image.fromarray(debug_slice).save(os.path.join(outdir, 'debug', f_seg[:-7] + str(i) + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slice and Dice the dataset')

    parser.add_argument('--rootdir', help='The root directory for the input images.')
    parser.add_argument('--outdir', help='Output directory for the processed images.')
    parser.add_argument('--debug', action='store_true', help='Display the Segmentations and Images in Debug Folder')
    args = parser.parse_args()
    slice_volumes(args.rootdir, args.outdir, args.debug)
