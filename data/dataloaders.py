from PIL import Image
import torch.utils.data as data
import nibabel as nib
import os
import re
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage.transform import resize
import albumentations as A

from .transformations import get_transform
from util.util import natural_sort

#################################################################################################################################################
DATASET_MODES = ['spinalcord', 'heart', 'prostate']
#################################################################################################################################################

class GenericDataset(data.Dataset):
    def __init__(self, rootdir, sites, datasetmode, phase='train', split_train=False, seed=0, target=False, data_ratio=1.0, batch_size=0):
        self.batch_size = batch_size
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')

        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'
        
        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.rootdir = rootdir
        self.sites = sites

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)]))
        else:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)]))
        
        self.all_segs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)]))
        
        assert len(self.all_imgs) == len(self.all_segs)
        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))

        if split_train:
            # make it repeatable on various calls so that train and test don't overlap.
            np.random.seed(seed)
            ratio = 0.3 if target else 0.8
            self.sampled_train_idx = np.random.choice(len(self.all_imgs), int(ratio * len(self.all_imgs)), replace=False)
        else:
            self.sampled_train_idx = np.arange(len(self.all_imgs))

        
        if phase == 'train' or phase == 'test' or phase=='train_no_color_T' or phase=='train_no_T':
            max_index = int(data_ratio*len(self.sampled_train_idx))
            self.all_imgs = self.all_imgs[self.sampled_train_idx[:max_index]]
            self.all_segs = self.all_segs[self.sampled_train_idx[:max_index]]
            #print(len(self.sampled_train_idx))
            #self.all_imgs = self.all_imgs[self.sampled_train_idx]
            #self.all_segs = self.all_segs[self.sampled_train_idx]
        
        elif phase== 'val': # chose validation data from the training set
            sampled_test_idx = [ele for ele in range(len(self.all_imgs)) if ele not in self.sampled_train_idx]
            self.all_imgs = self.all_imgs[sampled_test_idx]
            self.all_segs = self.all_segs[sampled_test_idx]
        else:
            raise Exception('Unrecognized phase.')

    def set_modulo(self, p_modulo):
        max_index = int(p_modulo * len(self.all_imgs))
        self.all_imgs = self.all_imgs[:max_index]
        self.all_segs = self.all_segs[:max_index]
        

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False
    
    def __getitem__(self, index):
        if self.batch_size != 0:
            index = index % len(self.all_imgs)
        
        img = np.load(os.path.join(self.rootdir, self.all_imgs[index])).astype(np.float32)
        seg = np.load(os.path.join(self.rootdir, self.all_segs[index]))

        transformed = self.augmenter(image=img, mask=seg)

        img = transformed['image']
        img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

        seg = transformed['mask']
        seg = torch.from_numpy(seg).to(torch.long)
        
        return img, seg

    
    def __len__(self):
        return max(len(self.all_imgs), self.batch_size)

class GenericDatasetPsudoLabels(data.Dataset):
    def __init__(self, rootdir, psudo_label_dir, sites, datasetmode, phase='train'):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')

        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'
        
        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.rootdir = rootdir
        self.sites = sites
        self.psudo_label_dir = psudo_label_dir

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)]))
        else:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)]))
        
        self.all_segs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)]))
        self.all_psudo_labels = np.array(natural_sort([f for f in os.listdir(psudo_label_dir) if seg_query in f and self.is_site(f)]))
        
        assert len(self.all_imgs) == len(self.all_segs)
        assert len(self.all_psudo_labels) == len(self.all_imgs)

        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False

    
    def __getitem__(self, index):
        img = np.load(os.path.join(self.rootdir, self.all_imgs[index])).astype(np.float32)
        seg = np.load(os.path.join(self.rootdir, self.all_segs[index]))
        psudo_labels = np.load(os.path.join(self.psudo_label_dir, self.all_psudo_labels[index])).astype(np.float32) # shape is n_class x 256 x 256
        # need to change it to match the shape of the segmentations
        psudo_labels = np.transpose(psudo_labels, [1, 2, 0]) # size is 256, 256, n_classes
        psudo_labels = resize(psudo_labels, seg.shape, order=0) # psudo_labels.shape[2]

        transformed = self.augmenter(image=img, masks=[seg, psudo_labels])

        img = transformed['image']
        img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

        seg = transformed['masks'][0]
        seg = torch.from_numpy(seg).to(torch.long)
        
        psudo_labels = transformed['masks'][1]
        psudo_labels = torch.from_numpy(psudo_labels).permute(2, 0, 1) # back to channel first

        return img, seg, psudo_labels

    
    def __len__(self):
        return len(self.all_imgs)

class GenericVolumeDataset(data.Dataset):
    def __init__(self, rootdir, sites, datasetmode, phase='train'):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')

        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'

        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.datasetmode = datasetmode
        self.rootdir = rootdir
        self.sites = sites

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)])
        else:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)])
        
        all_segs = natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)])
        
        assert len(all_imgs) == len(all_segs)

        grouped_imgs = {}
        grouped_segs = {}

        # group the slices based on patient
        for img, seg in zip(all_imgs, all_segs):
            if self.datasetmode == DATASET_MODES[0]:
                assert img.split('-')[:2] == seg.split('-')[:2]
                patient_name = '-'.join(img.split('-')[:2])
            elif self.datasetmode == DATASET_MODES[2]:
                assert re.split('-|_|\.', img)[:3] == re.split('-|_|\.', seg)[:3]
                patient_name = '-'.join(re.split('-|_|\.', img)[:3])
            else:
                assert img.split('_')[0] == seg.split('_')[0]
                patient_name = img.split('_')[0]
            
            if patient_name in grouped_imgs:
                grouped_imgs[patient_name].append(img)
                grouped_segs[patient_name].append(seg)

            else:
                grouped_imgs[patient_name] = [img]
                grouped_segs[patient_name] = [seg]
        
        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))

        self.all_imgs = [v for _ , v in grouped_imgs.items()]
        self.all_segs = [v for _ , v in grouped_segs.items()]

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False

    
    def __getitem__(self, index):
        img_names = self.all_imgs[index]
        seg_names = self.all_segs[index]

        out_imgs = []
        out_segs = []

        for i_n, s_n in zip(img_names, seg_names):    
            img = np.load(os.path.join(self.rootdir, i_n)).astype(np.float32)
            seg = np.load(os.path.join(self.rootdir, s_n))

            transformed = self.augmenter(image=img, mask=seg)

            img = transformed['image']
            img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

            seg = transformed['mask']
            seg = torch.from_numpy(seg).to(torch.long)
            
            out_imgs.append(img)
            out_segs.append(seg)
        
        out_imgs = torch.stack(out_imgs)
        out_segs = torch.stack(out_segs)

        return out_imgs, out_segs

    
    def __len__(self):
        return len(self.grouped_imgs)



class GenericVolumeDatasetPsudoLabels(data.Dataset):
    def __init__(self, rootdir, psudo_label_dir, sites, datasetmode, phase='train'):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')

        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'

        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.datasetmode = datasetmode
        self.rootdir = rootdir
        self.sites = sites
        self.psudo_label_dir = psudo_label_dir

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)])
        else:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)])
        
        all_segs = natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)])

        all_psudo_labels = natural_sort([f for f in os.listdir(psudo_label_dir) if seg_query in f and self.is_site(f)])
        
        assert len(all_imgs) == len(all_segs)

        grouped_imgs = {}
        grouped_segs = {}
        grouped_psudo_labels = {}

        # group the slices based on patient
        for img, seg, psudo_label in zip(all_imgs, all_segs, all_psudo_labels):
            if self.datasetmode == DATASET_MODES[0]:
                assert img.split('-')[:2] == seg.split('-')[:2]
                patient_name = '-'.join(img.split('-')[:2])
            elif self.datasetmode == DATASET_MODES[2]:
                assert re.split('-|_|\.', img)[:3] == re.split('-|_|\.', seg)[:3]
                patient_name = '-'.join(re.split('-|_|\.', img)[:3])
            else:
                assert img.split('_')[0] == seg.split('_')[0]
                patient_name = img.split('_')[0]
            
            if patient_name in grouped_imgs:
                grouped_imgs[patient_name].append(img)
                grouped_segs[patient_name].append(seg)
                grouped_psudo_labels[patient_name].append(psudo_label)

            else:
                grouped_imgs[patient_name] = [img]
                grouped_segs[patient_name] = [seg]
                grouped_psudo_labels[patient_name] = [psudo_label]
        
        # data augmentations
        self.augmenter = A.Compose(get_transform(phase))

        self.all_imgs = [v for _ , v in grouped_imgs.items()]
        self.all_segs = [v for _ , v in grouped_segs.items()]
        self.all_psudo_labels = [v for _, v in grouped_psudo_labels.items()]

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False

    
    def __getitem__(self, index):
        img_names = self.all_imgs[index]
        seg_names = self.all_segs[index]
        psudo_label_names = self.all_psudo_labels[index]

        out_imgs = []
        out_segs = []
        out_psudo_labels = []

        for i_n, s_n, p_n in zip(img_names, seg_names, psudo_label_names):    
            img = np.load(os.path.join(self.rootdir, i_n)).astype(np.float32)
            seg = np.load(os.path.join(self.rootdir, s_n))
            psudo_label = np.load(os.path.join(self.psudo_label_dir, p_n)).astype(np.float32)

            # change the shape to 256, 256, nclasses
            psudo_label = np.transpose(psudo_label, [1, 2, 0])
            psudo_label = resize(psudo_label, seg.shape, order=0)

            transformed = self.augmenter(image=img, masks=[seg, psudo_label])

            img = transformed['image']
            img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

            seg = transformed['masks'][0]
            seg = torch.from_numpy(seg).to(torch.long)

            psudo_label = transformed['masks'][1]
            psudo_label = torch.from_numpy(psudo_label).permute(2, 0, 1)
            
            out_imgs.append(img)
            out_segs.append(seg)
            out_psudo_labels.append(psudo_label)
        
        out_imgs = torch.stack(out_imgs)
        out_segs = torch.stack(out_segs)
        out_psudo_labels = torch.stack(out_psudo_labels)

        return out_imgs, out_segs, out_psudo_labels

    
    def __len__(self):
        return len(self.grouped_imgs)

class StyleDataset(Dataset):
    def __init__(self, dataroots, phase='train'):
        if not isinstance(dataroots, list):
            dataroots = [dataroots]

        all_imgs = []
        
        for dataroot in dataroots:
            all_imgs += [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataroot) for f in filenames if os.path.splitext(f)[1] in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']]
        
        
        self.all_imgs = all_imgs
        self.augmentor = A.Compose(get_transform(phase))
        print(len(self.all_imgs))

    def __getitem__(self, index):
        img = Image.open(self.all_imgs[index]).convert("L")
        img = np.array(img)

        img = self.augmentor(image=img)['image']/255.0
        img = img.astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0) # add channel

        img = 2.0 * img - 1.0

        return img

    def __len__(self,):
        return len(self.all_imgs)
