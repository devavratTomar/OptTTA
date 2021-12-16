import argparse
import os
import shutil
import pandas as pd

def pool_images(rootdir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    vendor_list = pd.read_csv(os.path.join(rootdir, '201014_M&Ms_Dataset_Information_-_opendataset.csv'), index_col='External code')
    img_dirs = ['train', 'test', 'validation']
    pooled_imgs_path = {
        'A': [],
        'B': [],
        'C': [],
        'D': []
    }

    for img_dir in img_dirs:
        path = os.path.join(rootdir, img_dir)
        if img_dir == 'train':
            path = os.path.join(path, 'Labeled')

        img_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        for img_name in img_names:
            pooled_imgs_path[vendor_list.loc[img_name]['Vendor']] += [os.path.join(path, img_name, f) for f in os.listdir(os.path.join(path, img_name))]

    for site, img_paths in pooled_imgs_path.items():
        for img_path in img_paths:
            new_name = 'site' + site + '-' + os.path.split(img_path)[1]

            shutil.copy(img_path, os.path.join(outdir, new_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Nii Volumes for Heart Dataset.')
    parser.add_argument('--rootdir', help='The root directory for the input images.')
    parser.add_argument('--outdir', help='Output directory for the processed images.')

    args = parser.parse_args()
    pool_images(args.rootdir, args.outdir)