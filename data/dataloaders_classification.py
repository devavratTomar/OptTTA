from PIL import Image
import PIL
import pandas as pd
import torch.utils.data as data
import nibabel as nib
import os
import re
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import albumentations as A

from .transformations import get_transform
from util.util import natural_sort
import pandas as pd

SITES = ['UDA-1', 'SONIC', 'UDA-2', 'MSK-2', 'MSK-1', 'MSK-3', 'MSK-4',
         '2018 JID Editorial Images', 'HAM10000', 'ISIC_2020_Vienna_part_1',
         'BCN_20000', 'ISIC_2020_Vienna_part2',
         'ISIC 2020 Challenge - MSKCC contribution',
         'Sydney (MIA / SMDC) 2020 ISIC challenge contribution',
         'BCN_2020_Challenge']

class SkinDataset(Dataset):
    def __init__(self, rootdir, sites, phase, split_train=False, seed=0) -> None:
        super().__init__()
        self.rootdir = rootdir
        self.all_imgs = [f for f in os.listdir(os.path.join(rootdir, 'images')) if f.endswith('.jpg')]
        self.meta_data = pd.read_csv(os.path.join(rootdir, 'metadata', 'ground_truth.csv'))

        self.classes = ['melanoma',
                        'nevus',
                        'dermatofibroma',
                        'basal cell carcinoma',
                        'vascular lesion',
                        'pigmented benign keratosis',
                        'actinic keratosis']

        self.classes_abv = ['mel',
                            'nv',
                            'df',
                            'bcc',
                            'vasc',
                            'bkl',
                            'akiec']

        self.augmenter = A.Compose(get_transform(phase))

        # first select images from the sites
        self.meta_data = self.meta_data[self.meta_data['dataset_name'].isin(sites)]
        filtered_imgs = self.meta_data['isic_id'].to_list()
        self.all_imgs =  np.array([f for f in self.all_imgs if os.path.splitext(f)[0] in filtered_imgs])

        assert len(self.all_imgs) == len(self.meta_data)

        if split_train:
            # make it repeatable on various calls so that train and test don't overlap.
            np.random.seed(seed)
            ratio = 0.9
            self.sampled_train_idx = np.random.choice(len(self.all_imgs), int(ratio * len(self.all_imgs)), replace=False)
        else:
            self.sampled_train_idx = np.arange(len(self.all_imgs))

        
        if phase == 'train' or phase == 'test' or phase=='train_no_color_T':
            self.all_imgs = self.all_imgs[self.sampled_train_idx]
        
        elif phase== 'val': # chose validation data from the training set
            sampled_test_idx = [ele for ele in range(len(self.all_imgs)) if ele not in self.sampled_train_idx]
            self.all_imgs = self.all_imgs[sampled_test_idx]
        else:
            raise Exception('Unrecognized phase.')

        nclass_samples = self.meta_data.groupby('diagnosis').count()['dataset_name']
        weights = 1.0/nclass_samples
        weights = weights/weights.sum()
        
        self.weights = []

        for c in self.classes:
            self.weights.append(weights.loc[c])


    def __getitem__(self, index):
        img_id, img_ext = os.path.splitext(self.all_imgs[index])

        img = Image.open(os.path.join(self.rootdir, 'images', img_id+img_ext))
        img = np.array(img).astype(np.float32)/255.0 # value is zero to one

        img = self.augmenter(image=img)['image']

        label = self.meta_data[self.meta_data['isic_id'] == img_id]['diagnosis'].iloc[0]
        label = torch.tensor(self.classes.index(label), dtype=torch.long)

        img = 2*img - 1.0 # w x h x n_channels
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label

    def __len__(self,):
        return len(self.all_imgs)