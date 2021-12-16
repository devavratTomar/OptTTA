import albumentations as A
import os
import argparse
from PIL import Image
import numpy as np
from util.util import getcolorsegs, ensure_dir
import torchvision.utils as tvu
import torch

STRING_TO_AUGMENTATIONS = {
    'brightness': A.RandomBrightness(always_apply=True, p=1.0),
    'gamma': A.RandomGamma(always_apply=True, p=1.0),
    'gaussianblur': A.GaussianBlur(always_apply=True),
    'contrast': A.RandomContrast(always_apply=True),
    'randomhorizontalflip': A.HorizontalFlip(),
    'randomverticalflip': A.VerticalFlip(),
    'randomrotate': A.RandomRotate90(),
    'randresizecrop': A.RandomResizedCrop(256, 256, scale=(0.5, 1.0), always_apply=True),
}

def generate_random_augs(root_img_path, root_seg_path, out_dir, augmentation_list, nsamples=16):
    """
    augmentation_list is a list of list of augmentations for generating segmentations
    """
    root_img = (255*np.load(root_img_path)).astype(np.uint8)
    root_seg = np.load(root_seg_path)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for augs in augmentation_list:
        policy_name = '_'.join(augs)
        ensure_dir(os.path.join(out_dir, policy_name))

        augmentor = [A.Resize(256, 256)]
        for aug in augs:
            if aug in STRING_TO_AUGMENTATIONS.keys():
                augmentor.append(STRING_TO_AUGMENTATIONS[aug])

        augmentor = A.Compose(augmentor)

        for i in range(nsamples):
            transformed = augmentor(image=root_img, mask=root_seg)
            img = transformed['image']
            seg = transformed['mask']

            # torch tensors
            img = torch.tensor(img/255.0)
            seg = torch.tensor(seg, dtype=torch.long)

            seg = getcolorsegs(seg)

            tvu.save_image(img, os.path.join(out_dir, policy_name, str(i) + '.png'))
            tvu.save_image(seg, os.path.join(out_dir, policy_name, str(i) + '_seg.png'))



parser = argparse.ArgumentParser("Generate dummy figures for paper diagram")

parser.add_argument('--root_img', type=str)
parser.add_argument('--root_seg', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--augs', type=str, help='ex. brightness_gamma,gaussianblur_contrast')

args = parser.parse_args()
aug_txt = args.augs

aug_list = aug_txt.split(',')
aug_list = [policy.split('_') for policy in aug_list]

print(aug_list)

generate_random_augs(args.root_img, args.root_seg, args.out_dir, aug_list)
