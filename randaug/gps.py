import torch
import os
import numpy as np
from .augmentations import *
import random
from data import GenericVolumeDataset
from networks.unet import UNet
import losses
from util.util import natural_sort
import pickle
import tqdm

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
    'GaussianBlur': [1, 3],
    'Contrast': [0.5, 1.9],
    'Brightness': [-0.5, 0.5],
    'Gamma': [0.2, 2.0],
    'RandomResizeCrop': [0.5, 1.0],
    'RandomHorizontalFlip': [0, 1],
    'RandomVerticalFlip': [0, 1],
    'RandomRotate': [0, 1]
}

def generate_sub_policies(all_augmentations, N, M, K):
    subpolicy_pool = []

    for i in range(K):
        ops_ids = sorted(random.sample(list(range(len(all_augmentations))), k=N))
        ops = []
        for ops_id in ops_ids:
            ops.append(all_augmentations[ops_id])
        
        if M != 0:
            mag = random.uniform(0, M)
        else:
            mag = 15

        if DummyAugmentor.__name__ in ops:
            ops.remove(DummyAugmentor.__name__)
            ops.append('RandomHorizontalFlip')
            ops.append('RandomVerticalFlip')
            ops.append('RandomRotate')
        
        subpolicy = []
        for op in ops:
            minv, maxv = minmax_values[op]
            m = (mag/30) * (maxv - minv) + minv
            subpolicy.append((op, m))

        subpolicy_pool.append(subpolicy)

    return subpolicy_pool

def generate_sub_policy_pool(output_path, save=True):
    """
    We create a pool of subpolicies with random magnitude
    """
    all_augmentations = [Gamma.__name__, GaussianBlur.__name__, Contrast.__name__, Brightness.__name__, Identity.__name__, RandomResizeCrop.__name__, DummyAugmentor.__name__]
    
    pool = []
    
    ## N = 3, M = 45, K = 500
    pool += generate_sub_policies(all_augmentations, N=3, M=30, K=500)
    pool += generate_sub_policies(all_augmentations, N=3, M=10, K=500)
    pool += generate_sub_policies(all_augmentations, N=3, M=0,  K=100)

    with open(os.path.join(output_path, 'PoolOfSubPolicies.txt'), 'w') as file:
        for line in pool:
            for p in line:
                file.write(str(p))
            file.write("\n")

    with open(os.path.join(output_path, 'PoolOfSubPolicies.pickle'), 'wb') as handle:
        pickle.dump(pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return pool

class GPS():
    def __init__(self, opt):
        self.opt = opt
        print("Test Time Data Augmentation")

    def initialize(self):
        #### Test Target Dataloader ####
        self.target_test_dataloader = GenericVolumeDataset(self.opt.dataroot, self.opt.target_sites, self.opt.dataset_mode, phase='test')
            
        ##### load pre-trained style manipulator and unet-segmentor
        self.unet = UNet(self.opt.ncolor_channels, self.opt.n_classes)
        self.load_pretrained()

        if self.opt.use_gpu:
            self.unet = self.unet.to(self.opt.gpu_id)
        
        # freeze weights
        self.freeze_weigths(self.unet)

        # eval mode
        self.unet.eval()

        # loss
        self.criterian_l1 = torch.nn.L1Loss()
        self.criterian_l2 = torch.nn.MSELoss()

        self.criterian_nuclear = losses.NuclearNorm()
        self.criterian_countour = losses.ContourRegularizationLoss(2)

        # runnign_vars, means
        self.running_means, self.running_vars = self.get_segmentor_bn_stats()
        self.bn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # 18 bn layers


        # generate a pool of sub policies
        generate_sub_policy_pool(self.opt.checkpoints_source_free_da)


    
    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    
    def load_pretrained(self):
        # unet
        latest = natural_sort([f for f in os.listdir(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models')) if f.endswith('.pth')])[-1]
        print('Loading checkpoint: ', latest)
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', latest), map_location='cpu')
        self.unet.load_state_dict(weights)

    def get_segmentor_bn_stats(self):
        running_means = []
        running_vars = []
        for l in self.unet.modules():
            if isinstance(l, torch.nn.BatchNorm2d):
                running_means.append(l.running_mean)
                running_vars.append(l.running_var)

        return running_means, running_vars

    @torch.no_grad()
    def smooth_loss(self, x, feats):
        loss = {}
        p = torch.softmax(x, dim=1)

        # entropy maps
        entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean() # E[p log p]
        loss['entropy'] = (100*entropy)*1.
        # match bn stats
        loss_bn = 0

        for i, (f, m, v) in enumerate(zip(feats, self.running_means, self.running_vars)):
            if i in self.bn_layers:
                # b x ch x h x w
                current_mean = f.mean(dim=(0, 2, 3))
                cuurent_var  = f.var(dim=(0, 2, 3))

                loss_bn += self.criterian_l2(current_mean, m) + self.criterian_l2(cuurent_var, v)

        loss['batchnorm_consistency'] = loss_bn*self.opt.alpha_1

        #loss['countour_consistency'] = 0.1*self.criterian_countour(x)
        loss['divergence']  = (-0.5*self.criterian_nuclear(p))*self.opt.alpha_2

        return loss
    
    @torch.no_grad()
    def ensemble_predictions(self, final_policy, tgt_vol):
        # final_policy is a list of subpolicy of type [(Brightness, 0.4), (Contrast, 0.6), ...]
        predictions_volume = []

        for sub_policy in final_policy:
            style_augmentors = [aug for aug in sub_policy if type(aug[0]).__name__ in STYLE_AUGMENTORS]
            spatial_augmentors = [aug for aug in sub_policy if type(aug[0]).__name__ in SPATIAL_AUGMENTORS]
            
            prediction_subpolicy = []

            for i in range(tgt_vol.size()[0]):
                aug_img = tgt_vol[i:(i+1)].clone()
                spatial_affines = []
                
                for (aug, value) in style_augmentors:
                    aug_img = aug(aug_img, torch.tensor(value, dtype=torch.float32)).detach()

                for (aug, value) in spatial_augmentors:
                    aug_img, affine = aug.test(aug_img, torch.tensor(value, dtype=torch.float32))
                    affine = affine.detach()
                    aug_img = aug_img.detach()
                    spatial_affines.append(affine)
                
                # get predictions on the augmented images of the ith slice
                pred = self.unet(aug_img.cuda()).detach().cpu()
                
                # invert affines in reverse order
                for (aug,value), affine in zip(reversed(spatial_augmentors), reversed(spatial_affines)):
                    inv_affine = aug.invert_affine(affine)
                    pred, inv_affine = aug.test(pred, torch.tensor(value, dtype=torch.float32), inv_affine)
                    pred = pred.detach()
                    inv_affine = inv_affine.detach()
                
                prediction_subpolicy.append(pred)
                # end of i loop
            
            prediction_subpolicy = torch.cat(prediction_subpolicy, dim=0) # same batch size as tgt_vol slices
            predictions_volume.append(prediction_subpolicy)
            # end of sub_policy loop

        predictions_volume = torch.stack(predictions_volume, dim=0) # size (T , n_sclices, n_classes, h, w)

        # take average across all sub_policies and compute uncertainty
        predictions_volume = torch.softmax(predictions_volume, dim=2) #T, n_slices, h, w
        prediction_labels = torch.argmax(predictions_volume, dim=2) #T, n_slices, h, w
    
        labels_frequency = torch.Tensor()
        for l in range(predictions_volume.shape[2]): #n_classes
            frequency_l = torch.sum(prediction_labels==l, dim=0, keepdim=True)/predictions_volume.shape[0] #1, n_slices, h, w
            labels_frequency = torch.cat([labels_frequency, frequency_l]) #n_classes, n_slices, h, w

        uncertainties_volume = -torch.mean(labels_frequency*torch.log(labels_frequency + 1e-6), dim=0) #n_slices, h, w

        predictions_volume = predictions_volume.mean(dim=0, keepdim=False)
        predictions_volume = torch.argmax(predictions_volume, dim=1) # size is (n_slices, h, w)

        return predictions_volume, uncertainties_volume

    def get_slice_index(self, img, threshold):
        out = []
        for i in range(img.size()[0]):
            tmp_img = img[i].clone() # don't spoil input image
            min_val = torch.quantile(tmp_img, 0.1)
            max_val = torch.quantile(tmp_img, 0.9)
            tmp_img[tmp_img<min_val] = min_val
            tmp_img[tmp_img>max_val] = max_val

            tmp_img = (tmp_img - min_val)/(max_val - min_val + 1e-8)
            if tmp_img.mean() > threshold:
                out.append(i)
        
        return out
    
    def save_pred_numpy(self, x, name):
        x = x.detach().cpu().numpy()
        np.save(os.path.join(self.opt.checkpoints_source_free_da, 'predictions', name), x)

    def save_uncertainty_numpy(self, x, name):
        x = x.detach().cpu().numpy()
        np.save(os.path.join(self.opt.checkpoints_source_free_da, 'uncertainties', name), x)

    def test_time_optimize(self, tgt_vol, final_policy_size=10, batch_size=16):
        with open(os.path.join(self.opt.checkpoints_source_free_da, 'PoolOfSubPolicies.pickle'), 'rb') as handle:
            policy_pool = pickle.load(handle)

        policy_loss = [] 

        for sub_policy in tqdm.tqdm(policy_pool):
            # apply the sub_policy to get the augmented image.
            # sub_policy is a list of augmentation ops.
            aug_volume = tgt_vol.clone()
            for (op, v) in sub_policy:
                augmentor = STRING_TO_CLASS[op]()
                aug_volume = augmentor(aug_volume, torch.tensor(v, dtype=torch.float32))

            # compute loss for this aug_volume
            if self.opt.use_gpu:
                aug_volume = aug_volume.cuda()

            avg_loss = 0
            # trick to fit in gpu
            for j in range(0, aug_volume.size()[0], batch_size):
                batched_img = aug_volume[j:(j+batch_size)]

                with torch.no_grad():
                    pred, feats = self.unet(batched_img, feats=True)
                    pred = pred.detach()

                loss = self.smooth_loss(pred, feats)
                avg_loss += loss['batchnorm_consistency'] + loss['entropy'] + loss['divergence']
            
            avg_loss = torch.mean(avg_loss)

            policy_loss.append(avg_loss.detach().cpu().item())
            # end of for loop

        best_policy_indices = np.argsort(policy_loss)[:final_policy_size]

        final_policy = []
        for i in best_policy_indices:
            sub_policy = policy_pool[i]
            opt_policy = []
            for op, v in sub_policy:
                augmentor = STRING_TO_CLASS[op]()
                opt_policy.append((augmentor, v))
            
            final_policy.append(opt_policy)

        pred, uncertainty = self.ensemble_predictions(final_policy=final_policy, tgt_vol=tgt_vol)
        
        return pred, uncertainty

    def launch(self):
        self.initialize()

        for iter, (img, seg) in enumerate(self.target_test_dataloader):
            all_imgs = self.target_test_dataloader.all_imgs[iter]
            all_segs = self.target_test_dataloader.all_segs[iter]

            if not isinstance(all_imgs, list):
                all_imgs = [all_imgs]
            
            if not isinstance(all_segs, list):
                all_segs = [all_segs]
            
            print('Predicting for image: ', all_imgs[0])

            if img.dim() != 4:
                img = img.unsqueeze(0)
                seg = seg.unsqueeze(0)

            # check effective batch size
            predictions = []
            uncertainties = []
            BATCH = img.size()[0]
            for i in range(0, img.shape[0], BATCH):
                pred, uncertainty = self.test_time_optimize(img[i:(i + BATCH)])
                predictions.append(pred)
                uncertainties.append(uncertainty)
            
            predictions = torch.cat(predictions, dim=0)
            uncertainties = torch.cat(uncertainties, dim=0)
            print(predictions.shape)
            for i in range(len(all_segs)):
                self.save_pred_numpy(predictions[i], all_segs[i])
                self.save_uncertainty_numpy(uncertainties[i], all_segs[i])