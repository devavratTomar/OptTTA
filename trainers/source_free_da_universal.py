import torch
import torch.nn.functional as tf
import os
from data import GenericDataset, GenericVolumeDataset
from torch.utils.data import DataLoader

import networks
from networks.unet import UNet
from networks import UniversalStyleManipulator

from util.util import overlay_segs
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm

from PIL import Image

class SourceFreeDomainAdaptorUniversal():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):
        #### Test Target Dataloader ####
        self.target_test_dataloader = GenericVolumeDataset(self.opt.dataroot, self.opt.target_sites, self.opt.dataset_mode, phase='test')
            
        ##### load pre-trained style manipulator and unet-segmentor
        self.unet = UNet(self.opt.ncolor_channels, self.opt.n_classes)
        self.style_manipulator = UniversalStyleManipulator()
        self.style_constraints = torch.tensor([1., 1., 1.], dtype=torch.float32).view(3, 1, 1)
        
        self.load_pretrained()

        if self.opt.use_gpu:
            self.unet = self.unet.to(self.opt.gpu_id)
            self.style_constraints = self.style_constraints.to(self.opt.gpu_id)
        
        # freeze weights
        self.freeze_weigths(self.unet)

        # eval mode
        self.unet.eval()

        # loss
        self.criterian_l1 = torch.nn.L1Loss()
        self.criterian_l2 = torch.nn.MSELoss()
        self.criterian_mumford_shah = losses.MumfordShahLoss()

        # runnign_vars, means
        self.running_means, self.running_vars = self.get_segmentor_bn_stats()
    
    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    
    def load_pretrained(self):
        # unet
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', 'Segmentor_1950000.pth'), map_location='cpu')
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


    def get_segmentor_bn_stats(self):
        running_means = []
        running_vars = []
        for l in self.unet.modules():
            if isinstance(l, torch.nn.BatchNorm2d):
                running_means.append(l.running_mean)
                running_vars.append(l.running_var)

        return running_means, running_vars

    def smooth_loss(self, x, feats, style):
        loss = {}

        p = torch.softmax(x, dim=1)

        # entropy maps
        entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean() # E[p log p]
        loss['entropy'] = entropy
        
        # match bn stats
        loss_bn = 0
        bn_layers = [0, 1, 2, 3]
        for i, (f, m, v) in enumerate(zip(feats, self.running_means, self.running_vars)):

            if i in bn_layers:
                # b x ch x h x w
                current_mean = f.mean(dim=(0, 2, 3))
                cuurent_var  = f.var(dim=(0, 2, 3))

                loss_bn += self.criterian_l2(current_mean, m) + self.criterian_l2(cuurent_var, v)

        loss_bn = loss_bn

        loss['batchnorm_consistency'] = 0.1*loss_bn

        loss['style_constraints'] = (self.style_constraints * (style**2).mean(dim=1)).sum() # shape of style is n_styles x batch x channels, style_constraints is n_styles x 1 x 1

        return loss
    

    def test_time_optimize(self, target_image):
        ## perform test time optimization

        # manipulate style to minimize the smoothness loss on Unet
        iter_style = torch.zeros(3, *target_image.size()[0:2], dtype=torch.float32, device=target_image.device).requires_grad_(True)

        ## optimizer
        optimizer = torch.optim.Adam([{'params': [iter_style]}], lr=0.01, betas=(0.9,0.999), eps=1e-8)
        
        schedular = self.get_schedular(optimizer, linear_steps=0, const_setps=self.opt.n_steps, cosine_steps=0)

        ## visualizations
        series_visuals = []
        series_visuals_imgs = []
        series_losses = []

        # Pre-visual
        seg_pred = self.unet(target_image)
        series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True))
        series_visuals_imgs.append(target_image.detach())

        ## we update the style code to fix the segmentation output of UNet
        for i in tqdm.tqdm(range(self.opt.n_steps)):
            optimizer.zero_grad()

            ## reconstrut image using style manipulator
            syn_img = self.style_manipulator(target_image, iter_style)
            seg_pred, feats = self.unet(syn_img, feats=True)

            series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True))
            series_visuals_imgs.append(syn_img.detach())

            ## smoothness loss
            losses = self.smooth_loss(seg_pred, feats, iter_style)
            loss = sum([v for v in losses.values()])
            loss.backward()
            optimizer.step()
           

            if i < self.opt.n_steps - 1:
                schedular.step()

            # for visulaization
            series_losses.append([v.detach().item() for v in losses.values()] + [loss.detach().item()])
        
        ## process the data
        series_losses = torch.tensor(series_losses)
        series_visuals = tvu.make_grid(torch.cat(series_visuals, dim=0), nrow=4)[0] # make_grid makes the channel size to 3
        series_visuals_imgs = tvu.make_grid(torch.cat(series_visuals_imgs, dim=0), nrow=4)
        series_visuals = overlay_segs(series_visuals_imgs, series_visuals)


        return torch.argmax(seg_pred, dim=1), series_losses, series_visuals, series_visuals_imgs

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
        
        x = torch.cat([y.detach().cpu() for y in x], dim=0)

        tvu.save_image(x, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', img_name + '.png'))

    def save_pred_numpy(self, x, name):
        x = x.detach().cpu().numpy()
        np.save(os.path.join(self.opt.checkpoints_source_free_da, 'predictions', name), x)

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

            if self.opt.use_gpu:
                img = img.to(self.opt.gpu_id)

            # check effective batch size
            predictions = []
            BATCH = 1
            for i in range(0, img.shape[0], BATCH):                
                pred, losses, visuals, visuals_imgs = self.test_time_optimize(img[i:(i + BATCH)])

                if self.opt.n_steps > 0:
                    self.save_plots(losses, all_imgs[i], ['entropy', 'batchnorm_consistency', 'style_constraints', 'total'])
                
                self.save_visuals(visuals, 'segmentations_iters_' + all_imgs[i])
                self.save_visuals([img[i:(i + BATCH)].repeat([1, 3, 1, 1]).clamp(-1, 1) * 0.5 + 0.5, overlay_segs(img[i:(i + BATCH)], seg[i:(i + BATCH)]), overlay_segs(img[i:(i + BATCH)], pred)],
                                'predictions_' + all_imgs[i])

                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0)

            for i in range(len(all_segs)):
                self.save_pred_numpy(predictions[i], all_segs[i])                 


