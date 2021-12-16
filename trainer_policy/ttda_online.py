import torch
from torch._C import device
import torch.nn.functional as tf
import os
from data import GenericVolumeDatasetPsudoLabels, GenericVolumeDataset
from torch.utils.data import DataLoader
import random

import kornia as K
import networks
from networks.unet import UNet

from util.util import overlay_segs, getcolorsegs, natural_sort, ensure_dir
from util import MetricTracker
from util import IterationCounter, Visualizer, MetricTracker, segmentation_score_stats, dice_coef_multiclass

import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm

import itertools

from PIL import Image

DEBUG = True

class TTDAOnline():
    def __init__(self, opt):
        self.opt = opt
        print("Test Time Data Augmentation")

    def initialize(self):
        #### Test Target Dataloader ####
        self.target_test_dataloader = GenericVolumeDatasetPsudoLabels(self.opt.dataroot, self.opt.psudo_root, self.opt.target_sites, self.opt.dataset_mode, phase='test')
        self.target_train_dataloader = GenericVolumeDatasetPsudoLabels(self.opt.dataroot, self.opt.psudo_root, self.opt.target_sites, self.opt.dataset_mode, phase='train')

        ##### perform n updates using gradient descent on the target image
        
        print("Number of iterations : ", self.opt.n_steps)

        self.criterian_ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.criterian_dc  = K.losses.DiceLoss()
        self.criterian_sce = losses.SCELoss(alpha=0.1, beta=1, num_classes=self.opt.n_classes)
        self.criterian_countour = losses.ContourRegularizationLoss(2)

        ##### load pre-trained style manipulator and unet-segmentor
        self.unet = UNet(self.opt.ncolor_channels, self.opt.n_classes)
        self.load_pretrained()
        
        ## optimizers, schedulars
        self.optimizer = self.get_optimizers()

        if self.opt.use_gpu:
            self.unet = self.unet.to(self.opt.gpu_id)

        self.unet.train()

        self.is_first = True

    def get_optimizers(self):
        params = self.get_params()
        params_group = [
            {
                'params': params
            },
        ]

        optimizer = torch.optim.SGD(params_group, lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        # optimizer = torch.optim.AdamW(params_group, lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        # schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.opt.total_nimgs // (self.opt.batch_size), 0.1*self.opt.lr)
        # schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.total_nimgs)
        return optimizer

    
    def freeze_weigths(self, net):
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for param in m.parameters():
                    param.requires_grad_(False)

    def get_slice_index(self, psudolabels):
        out = []
        for i in range(psudolabels.size()[0]):
            tmp_img = psudolabels[i].clone()
            tmp_img = torch.argmax(tmp_img, dim=0)
            if tmp_img.sum() != 0:
                out.append(i)

        
        return out

    @torch.no_grad()
    def reset_batchnorm_params(self, target_img):
        self.unet.train()

        for m in self.unet.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                # print("Old means:%f Old Variance: %f" %(m.running_mean.mean(), m.running_var.mean()))
                m.reset_running_stats()

        print("\nUpdating Batch norm stats according to the target domain.\n")
        pre_bn_feats = {}

        for i in range(target_img.size()[0]):
            img = target_img[i:(i+1)]
            _, feats = self.unet(img, feats=True)

            for j, feat in enumerate(feats):
                if j in pre_bn_feats:
                    pre_bn_feats[j].append(feat)
                else:
                    pre_bn_feats[j] = [feat]
        
        running_means = {}
        running_vars =  {}
        for i in pre_bn_feats:
            running_means[i] = torch.cat(pre_bn_feats[i], dim=0).mean(dim=(0, 2, 3))
            running_vars[i]  = torch.cat(pre_bn_feats[i], dim=0).var(dim=(0, 2, 3))


        i = 0
        for m in self.unet.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = running_means[i].to(device=self.opt.gpu_id)
                m.running_var = running_vars[i].to(device=self.opt.gpu_id)
                i += 1
        
        
        # for epoch in tqdm.tqdm(range(1000)):
        #     images = target_img[random.choices(slice_indices, k=self.opt.sample_size)]
        #     if self.opt.use_gpu:
        #         images = images.to(device=self.opt.gpu_id)
        #     self.unet(images)

        # for m in self.unet.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         print("New means: %f New variance: %f" % (m.running_mean.mean(), m.running_var.mean()))
        self.unet.eval()


    def load_pretrained(self):
        # unet
        latest = natural_sort([f for f in os.listdir(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models')) if f.endswith('.pth')])[-1]
        print('Loading checkpoint: ', latest)
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', latest), map_location='cpu')
        self.unet.load_state_dict(weights)
    



    def get_threshold(self, target_psudo_labels):
        class_threshold = {
        }

        for c in range(1, self.opt.n_classes):
            for psudo_label in target_psudo_labels:
                probs_hard_labels, hard_labels = torch.max(psudo_label, dim=0)
                class_probs = probs_hard_labels[hard_labels == c]

                if c not in class_threshold:
                    class_threshold[c] = class_probs.flatten()
                else:
                    class_threshold[c] = torch.cat([class_threshold[c], class_probs.flatten()])

        for c in class_threshold:
            class_threshold[c] = class_threshold[c].mean() #+ class_threshold[c].std()
        
        print(class_threshold)
        class_threshold[0] = 0.8
        return class_threshold




    def get_params(self):
        return list(self.unet.parameters())
        # params = []
        # for m in self.unet.modules():
        #     if not isinstance(m, torch.nn.BatchNorm2d):
        #         params += list(m.parameters())
        
        # return params
    

    def psudo_labelling_loss(self, logits, probs):
        p = torch.softmax(logits, dim=1)
        
        ## using ttda
        probs_hard_labels, hard_labels = torch.max(probs, dim=1)

        ## using current model
        model_probs_hard_labels, model_hard_labels = torch.max(p.detach(), dim=1)

        # # entropy maps
        # entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean()


        loss = 0
        n_pixels = 0

        tmp_loss = 0
        pixels_of_interest_global = torch.zeros_like(probs_hard_labels, dtype=torch.bool)
        for c in range(self.opt.n_classes):
            if probs_hard_labels[hard_labels == c].sum() != 0:
                # train with only confident pixels from TTDA
                threshold = 0.8 #self.threshold[c]

                pixels_of_interest = torch.bitwise_and((probs_hard_labels > threshold), (hard_labels == c))
                
                pl = hard_labels[pixels_of_interest]
                logits_masked = logits.permute(0,2,3,1)[pixels_of_interest]
                tmp_loss += self.criterian_sce(logits_masked, pl).sum()
                n_pixels += len(logits_masked)

                pixels_of_interest_global = torch.bitwise_or(pixels_of_interest_global, pixels_of_interest)


        loss += tmp_loss/n_pixels
                
        # add pixels which are already predicted well by the model
        model_pixels_of_interest = (model_probs_hard_labels > 0.8)
        model_pixels_of_interest = torch.bitwise_and(model_pixels_of_interest, ~pixels_of_interest_global) 
        if model_pixels_of_interest.sum() > 0:
            model_pl = model_hard_labels[model_pixels_of_interest]
            model_logits_masked = logits.permute(0,2,3,1)[model_pixels_of_interest]
            loss += self.criterian_sce(model_logits_masked, model_pl).mean()


        # loss_consistency = self.criterian_countour(torch.softmax(logits, dim=1))
        return loss

    def get_circular_index(self, i, n_samples, n_index):
        right = []
        left = []
        for j in range(i, i + n_samples//2):
            right.append( j % n_index)
        for j in range(i - (n_samples - n_samples//2), i):
            left.append( j % n_index)

        return left + right


    def train_online(self, target_image, target_psudo_labels):
        print(target_image.shape, target_psudo_labels.shape)
        
        ##### the already saved batch norm statistics will get updated very slowing (momentum of 0.9) to the current
        ##### batch norm stats with fine tuning. So we first do forward pass just to make the batch norm stats consistent on the new domain.
        # self.reset_batchnorm_params(target_image)
        # self.freeze_weigths(self.unet)

        # metrics tracker
        self.metric_tracker = MetricTracker()
        ## metrics
        self.dice_coef = dice_coef_multiclass

        self.threshold = self.get_threshold(target_psudo_labels)

        slices_index = self.get_slice_index(target_psudo_labels)

        lr = self.opt.lr if self.is_first else self.opt.lr

        self.update_lr(lr)


        progress_bar = tqdm.tqdm(range(self.opt.n_steps))
        for _ in progress_bar:
            for sample_index in slices_index:
                self.optimizer.zero_grad()
                imgs = target_image[sample_index:(sample_index+1)]
                psudo_labels = target_psudo_labels[sample_index:(sample_index+1)]

                predict = self.unet(imgs)

                loss = self.psudo_labelling_loss(predict, psudo_labels)
                loss.backward()
                
                self.optimizer.step()
            progress_bar.set_description("ce loss: %f" % loss)


        ### testing on the target image
        # with torch.no_grad():
        #     predictions = []
        #     for i in range(0, target_image.size()[0], self.opt.sample_size):
        #         imgs = target_image[i: (i+self.opt.sample_size)]
        #         predict = self.unet(imgs).detach()
        #         predictions.append(torch.argmax(predict, dim=1))
        
        with torch.no_grad():
            predictions = []
            slices_index = list(range(min(slices_index), max(slices_index) + 1))
            for i in range(0, target_image.size()[0]):
                if i in slices_index:
                    imgs = target_image[i: (i+1)]
                    predict = self.unet(imgs).detach()
                    predict = torch.argmax(predict, dim=1)
                else:
                    predict = torch.zeros(1, *target_image.size()[2:], dtype=torch.long, device=target_image.device)
                
                predictions.append(predict)

        self.is_first = False
        return torch.cat(predictions, dim=0)
    

    def save_pred_numpy(self, x, name):
        x = x.detach().cpu().numpy()
        np.save(os.path.join(self.opt.checkpoints_source_free_da, 'predictions', name), x)

    def visualize_segmentations(self, imgs, segs, predictions, psudo_labels, name):
        psudo_labels = torch.argmax(psudo_labels.detach().cpu(), dim=1)
        viz = [overlay_segs(imgs.detach().cpu(), segs), overlay_segs(imgs.detach().cpu(), psudo_labels), overlay_segs(imgs.detach().cpu(), predictions.detach().cpu()),]
        viz = torch.cat(viz, dim=0)
        viz = tvu.make_grid(viz, nrow=imgs.size()[0])
        tvu.save_image(viz, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', name + '.png'))


    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def launch(self):
        self.initialize()
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'predictions'))
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'visuals'))

        for i, (img, seg, psudo_label) in enumerate(self.target_test_dataloader):
            slices_seg_names = self.target_test_dataloader.all_segs[i]
            print("Processing image: ", slices_seg_names[0])
            if self.opt.use_gpu:
                img = img.to(device=self.opt.gpu_id)
                psudo_label = psudo_label.to(device=self.opt.gpu_id)

            predictions = self.train_online(img, psudo_label)
            
            self.visualize_segmentations(img, seg, predictions, psudo_label, slices_seg_names[0])

            for j in range(predictions.size()[0]):
                self.save_pred_numpy(predictions[j], slices_seg_names[j])