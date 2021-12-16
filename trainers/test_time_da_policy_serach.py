import torch
import torch.nn.functional as tf
import os
from data import GenericDataset, GenericVolumeDataset
from torch.utils.data import DataLoader
import shutil

import networks
from networks.unet import UNet
from trainers.differentiable_augmentations import Gamma, Contrast, Brightness, RandomResizeCrop, GaussianBlur, RandomHorizontalFlip, RandomRotate, RandomVerticalFlip

from util.util import overlay_segs
from util.util import natural_sort, ensure_dir
import torchvision.utils as tvu
import torchvision.transforms.functional as tvf
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm
import random

from PIL import Image, ImageOps 

FAST = False

class TestTimePolicySearch():
    def __init__(self, opt):
        self.opt = opt
        print("Test Time Poicy Search for Augmentations")

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

        # self.batch_norm_params = self.get_bn_parameters()
    
    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    
    def load_pretrained(self):
        # unet
        latest = natural_sort([f for f in os.listdir(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models')) if f.endswith('.pth')])[-1]
        print('Loading checkpoint: ', latest)
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', latest), map_location='cpu')
        self.unet.load_state_dict(weights)

    def get_schedular(self, optim, linear_steps, const_setps, cosine_steps):
        def lr_multiplier(step):
            if step < linear_steps:
                mult = step/linear_steps
            elif linear_steps <= step < linear_steps + const_setps:
                mult = 1.0
            else:
                mult = 0.5*(1.0 + math.cos(math.pi*(step - linear_steps - const_setps)/(cosine_steps + 1)))

            return mult

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_multiplier)

    def get_bn_parameters(self):
        params = []

        for l in self.unet.modules():
            if isinstance(l, torch.nn.BatchNorm2d):
                params.extend(list(l.parameters()))

        return params


    def get_segmentor_bn_stats(self):
        running_means = []
        running_vars = []
        for l in self.unet.modules():
            if isinstance(l, torch.nn.BatchNorm2d):
                running_means.append(l.running_mean)
                running_vars.append(l.running_var)

        return running_means, running_vars

    def smooth_loss(self, x, feats):
        loss = {}

        p = torch.softmax(x, dim=1)

        # entropy maps
        entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean() # E[p log p]
        loss['entropy'] = 100*entropy
        
        # match bn stats
        loss_bn = 0

        for i, (f, m, v) in enumerate(zip(feats, self.running_means, self.running_vars)):
            if i in self.bn_layers:
                # b x ch x h x w
                current_mean = f.mean(dim=(0, 2, 3))
                cuurent_var  = f.var(dim=(0, 2, 3))

                loss_bn += self.criterian_l2(current_mean, m) + self.criterian_l2(cuurent_var, v)

        loss['batchnorm_consistency'] = loss_bn

        # loss['countour_consistency'] = 0.1*self.criterian_countour(x)

        loss['divergence'] = -0.5*self.criterian_nuclear(p)

        return loss
    
    @torch.no_grad()
    def ensemble_predictions(self, augmentors, tgt_img, k=32):
        # tgt_img is batch x ch x h x w
        pred_augs = []
        pred_vol = []

        for j in range(tgt_img.size()[0]):
            pred_logits = []
            for i in range(k):
                # change the style of the images
                aug_img = tgt_img[j:(j+1)]
                for augmentor in augmentors['style']:
                    aug_img = augmentor(aug_img).detach()
                
                spatial_affines = []
                for augmentor in augmentors['spatial']:
                    aug_img, affine = augmentor.test(aug_img)
                    spatial_affines.append(affine)
                    aug_img = aug_img.detach()

                # visualizations
                if i% (k//4) == 0:
                    pred_augs.append(aug_img.detach())
                
                # predict on Unet
                pred = self.unet(aug_img).detach()
                
                # revert back to original spatial format
                for augmentor, affine in zip(reversed(augmentors['spatial']), reversed(spatial_affines)):
                    inv_affine = augmentor.invert_affine(affine)
                    pred, _ = augmentor.test(pred, inv_affine)
                    pred = pred.detach()
                
                pred_logits.append(pred.detach())
            
            # ensemble the predictions
            pred_logits = torch.cat(pred_logits, dim=0)
            pred_softmax = torch.softmax(pred_logits, dim=1)
            avg_pred = pred_softmax.mean(dim=0, keepdim=True)
            pred_vol.append(torch.argmax(avg_pred, dim=1))
        
        pred_vol = torch.cat(pred_vol, dim=0).detach().cpu()
        pred_augs = torch.cat(pred_augs, dim=0).detach().cpu()
        return pred_vol, pred_augs

    def delete_augmentors_checkpoints(self):
        shutil.rmtree(os.path.join(self.opt.checkpoints_source_free_da, 'augmentors_checkpoints'))
    
    def checkpoint_augmentors(self, epoch, name, augmentor):
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'augmentors_checkpoints'))
        torch.save(augmentor.state_dict(), os.path.join(self.opt.checkpoints_source_free_da, 'augmentors_checkpoints', name + '_' + str(epoch) + '.pth'))

    def load_augmentors(self, epoch, name, augmentor):
        weights = torch.load(os.path.join(self.opt.checkpoints_source_free_da, 'augmentors_checkpoints', name + '_' + str(epoch) + '.pth'))
        augmentor.load_state_dict(weights)
        return augmentor

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
    
    def test_time_optimize(self, target_image, sample_size=12):
        ## perform test time optimization
        ## differentiable data augmentations
        random_gamma = Gamma(self.opt.ncolor_channels)
        random_contrast = Contrast(self.opt.ncolor_channels)
        random_brightness = Brightness(self.opt.ncolor_channels)
        random_affine = RandomResizeCrop()
        random_gaussian = GaussianBlur()

        ## non differentiable
        random_horizontalflip = RandomHorizontalFlip()
        random_verticalflip = RandomVerticalFlip()
        random_rotate = RandomRotate()

        if self.opt.use_gpu:
            random_gamma = random_gamma.to(device=self.opt.gpu_id)
            random_contrast = random_contrast.to(device=self.opt.gpu_id)
            random_brightness = random_brightness.to(device=self.opt.gpu_id)
            random_affine = random_affine.to(device=self.opt.gpu_id)
            random_gaussian = random_gaussian.to(device=self.opt.gpu_id)
            random_horizontalflip = random_horizontalflip.to(device=self.opt.gpu_id)
            random_verticalflip = random_verticalflip.to(device=self.opt.gpu_id)
            random_rotate = random_rotate.to(device=self.opt.gpu_id)

        ## parameters to optimize
        opt_params = list(random_contrast.parameters()) + list(random_affine.parameters()) + list(random_gaussian.parameters())\
                   #+ list(random_brightness.parameters()) # list(random_gamma.parameters())
        
        ## optimizer
        optimizer = torch.optim.AdamW([{'params': opt_params}], lr=0.005, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)
        schedular = self.get_schedular(optimizer, linear_steps=0, const_setps=self.opt.n_steps, cosine_steps=0)

        ## visualizations
        series_visuals = []
        series_visuals_imgs = []
        series_losses = []

        # remove all black images from slices
        slice_indices = self.get_slice_index(target_image, 0.2)
        print(slice_indices)

        # Pre-visual
        seg_pred = self.unet(target_image).detach().cpu()
        series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True))
        series_visuals_imgs.append(target_image.detach().cpu())

        ## get optimal test time augmentations by training n_steps
        # target_duplicates = target_image.repeat(sample_size, 1, 1, 1) # create sample_size copies of the target_image
        
        loss_tracker = []
        window_size = 50

        for i in tqdm.tqdm(range(self.opt.n_steps)):
            optimizer.zero_grad()
            batched_targets = target_image[random.choices(slice_indices, k=sample_size)]

            aug_imgs = random_rotate(random_verticalflip(random_horizontalflip(random_affine(random_contrast(random_gaussian(batched_targets))))))
            seg_pred, feats = self.unet(aug_imgs, feats=True)

            ## smoothness loss
            losses = self.smooth_loss(seg_pred, feats)
            loss = sum([v for v in losses.values()])
            loss.backward()
            optimizer.step()
            schedular.step()
            series_losses.append([v.detach().cpu().item() for v in losses.values()] + [loss.detach().cpu().item()])

            # for visulaization
            if i % 50 == 0:
                series_visuals.append(torch.argmax(seg_pred.detach().cpu(), dim=1, keepdim=True))
                series_visuals_imgs.append(aug_imgs.detach().cpu())
            
            if FAST:
                # save the current augmentors
                self.checkpoint_augmentors(i, 'random_gamma', random_gamma)
                self.checkpoint_augmentors(i, 'random_contrast', random_contrast)
                self.checkpoint_augmentors(i, 'random_brightness', random_brightness)
                self.checkpoint_augmentors(i, 'random_affine', random_affine)
                
                # stopping criterian
                loss_tracker.append(loss.detach().cpu().item())
                if i > window_size:
                    threshold = np.max(loss_tracker[(i-window_size):i]) # does not include the ith value appended above
                    if loss_tracker[i] > threshold:
                        break
        
        if FAST:
            # find the iteration of the best loss
            best_aug_index = np.argmin(loss_tracker)
            print('Best index: ', best_aug_index)
            # load the state dicts
            random_gamma = self.load_augmentors(best_aug_index, 'random_gamma', random_gamma)
            random_contrast = self.load_augmentors(best_aug_index, 'random_contrast', random_contrast)
            random_brightness = self.load_augmentors(best_aug_index, 'random_brightness', random_brightness)
            random_affine = self.load_augmentors(best_aug_index, 'random_affine', random_affine)

        ## using learned policy generate augmented views:
        augmentors = {'style':[random_gaussian, random_contrast], 'spatial': [random_affine, random_horizontalflip, random_verticalflip, random_rotate]}
        final_predictions, pred_aug_imgs = self.ensemble_predictions(augmentors, target_image)

        ## process the data
        series_losses = torch.tensor(series_losses)
        
        series_visuals = tvu.make_grid(torch.cat(series_visuals, dim=0), nrow=4)[0] # make_grid makes the channel size to 3
        series_visuals_imgs = tvu.make_grid(torch.cat(series_visuals_imgs, dim=0), nrow=4)
        series_visuals = overlay_segs(series_visuals_imgs, series_visuals)

        pred_aug_visuals = tvu.make_grid(0.5*pred_aug_imgs + 0.5, nrow=4) # make_grid makes the channel size to 3

        return final_predictions, series_losses, series_visuals, series_visuals_imgs, pred_aug_visuals

    def save_plots(self, x, img_name, legend):
        x = x.detach().cpu().numpy()

        for i in range(x.shape[1]):
            plt.plot(np.arange(x.shape[0]), x[:, i])

        plt.legend(legend)
        plt.savefig(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'losses_' + img_name + '.png'))
        plt.clf()


    def save_visuals(self, x, img_name):
        if not isinstance(x, list):
            x = [x]
        
        n_rows = x[0].size(0)
        x = torch.cat([y.detach().cpu() for y in x], dim=0)
        x = tvu.make_grid(x, nrow=n_rows)

        tvu.save_image(x, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', img_name + '.png'))

    def save_pred_numpy(self, x, name):
        x = x.detach().cpu().numpy()
        np.save(os.path.join(self.opt.checkpoints_source_free_da, 'predictions', name), x)

    def launch(self):
        self.initialize()

        for iter, (img, seg) in enumerate(self.target_test_dataloader):
            print(img.shape)
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

            if self.opt.use_gpu:
                img = img.to(self.opt.gpu_id)

            # check effective batch size
            predictions = []
            BATCH = img.size()[0]
            
            for i in range(0, img.shape[0], BATCH):
                pred, losses, visuals, visuals_imgs, aug_views = self.test_time_optimize(img[i:(i + BATCH)])
                if self.opt.n_steps > 0:
                    self.save_plots(losses, all_imgs[i], ['entropy', 'batchnorm_consistency', 'div', 'total'])
                
                self.save_visuals(visuals, 'segmentations_iters_' + all_imgs[i])
                self.save_visuals([img[i:(i + BATCH)].repeat([1, 3, 1, 1]).clamp(-1, 1) * 0.5 + 0.5, overlay_segs(img[i:(i + BATCH)], seg[i:(i + BATCH)]), overlay_segs(img[i:(i + BATCH)], pred)],
                                'predictions_' + all_imgs[i])
                
                self.save_visuals(aug_views, 'augmented_views_' + all_imgs[i])

                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0)

            for i in range(len(all_segs)):
                self.save_pred_numpy(predictions[i], all_segs[i])                 


