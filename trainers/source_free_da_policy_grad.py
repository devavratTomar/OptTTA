import torch
import torch.nn.functional as tf
import os
from data import GenericDataset, GenericVolumeDataset
from torch.utils.data import DataLoader

import networks
from networks.unet import UNet
from networks import UniversalStyleManipulator, UniversalSpatialManipulator

from util.util import overlay_segs
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm

from PIL import Image

class SourceFreeDomainAdaptorUniversalPolicyGradient():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):
        #### Test Target Dataloader ####
        self.target_test_dataloader = GenericVolumeDataset(self.opt.dataroot, self.opt.target_sites, self.opt.dataset_mode, phase='test')
            
        ##### load pre-trained style manipulator and unet-segmentor
        self.unet = UNet(self.opt.ncolor_channels, self.opt.n_classes)
        self.style_manipulator = UniversalStyleManipulator()
        self.spatial_manipulator = UniversalSpatialManipulator()
        
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

        # runnign_vars, means
        self.running_means, self.running_vars = self.get_segmentor_bn_stats()
    
    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    
    def load_pretrained(self):
        # unet
        print('Loading pretrained model at: ', os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', 'Segmentor_250000.pth'))
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', 'Segmentor_250000.pth'), map_location='cpu')
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

    def smooth_loss(self, x, feats):
        loss = {}
        p = torch.softmax(x, dim=1)

        # entropy maps
        entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean() # E[p log p]
        loss['entropy'] = entropy
        
        # match bn stats
        loss_bn = 0
        bn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i, (f, m, v) in enumerate(zip(feats, self.running_means, self.running_vars)):

            if i in bn_layers:
                # b x ch x h x w
                current_mean = f.mean(dim=(0, 2, 3))
                cuurent_var  = f.var(dim=(0, 2, 3))

                loss_bn += self.criterian_l2(current_mean, m) + self.criterian_l2(cuurent_var, v)

        loss_bn = loss_bn

        loss['batchnorm_consistency'] = 0.1*loss_bn

        # uniformity of probability gent
        # avg_p = p.mean(dim=[0, 2, 3]) # avg along the pixels dim h x w and batch
        # divergence = torch.sum(avg_p * torch.log(avg_p + 1e-6))
        # loss['divergence'] = 0.1*divergence
        
        return loss

    def get_homographic_mat(self, A):
        H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
        H[..., -1, -1] += 1.0

        return H

    
    @torch.no_grad()
    def get_predictions_spatial_transform(self, target_imgs, params_spatial, ensemble_size):
        batch_size, _,  n_channels, height, width = target_imgs.size()
        
        spatials = params_spatial['bias'] + params_spatial['range']*torch.rand(batch_size, ensemble_size, 6, device='cpu')
        
        inv_spatials = self.get_homographic_mat(spatials.view(batch_size*ensemble_size, 2, 3))
        inv_spatials = torch.inverse(inv_spatials)[:, :2, :3]
        inv_spatials = inv_spatials.view(spatials.size())

        target_imgs = target_imgs.repeat([1, ensemble_size, 1, 1, 1]) #batch x ensemble_size x ch x h x w

        modified_imgs = self.spatial_manipulator(target_imgs, spatials).detach()
        predictions = self.unet(modified_imgs.view(batch_size*ensemble_size, n_channels, height, width)).detach()

        # get back to the original space
        predictions_orig = self.spatial_manipulator(predictions, inv_spatials).detach()
        return predictions_orig

        # size is batch x sample_size x 6
        inv_spatial = torch.inverse()
    
    @torch.no_grad()
    def get_ensemble_predictions(self, params_style, params_spatial, target_imgs, ensemble_size):
        assert target_imgs.dim() == 4 # batch x ch x h x w

        #######################################
        target_imgs = target_imgs.unsqueeze(1)

        ####################################### Save synthetic images and predictions
        styled_imgs = []
        predictions = []
        
        ####################################### first apply style manipulation to generate ensemble_size images ###########################
        batch_size, _,  n_channels = target_imgs.size()[0:3]

        # don't load the GPU yet
        styles = params_style['bias'] + params_style['range']*torch.rand(batch_size, ensemble_size, n_channels, 3, device='cpu')

        for i in range(ensemble_size):
            style = styles[:, i:(i+1), ...].cuda()
            styled_imgs += [self.style_manipulator(target_imgs, style).detach()]  # shape is batch x sample_size x n_channels x h x w

        


        


        preds = self.unet(syn_imgs).softmax(dim=1) # (n_images.sample_size) x n_classes x h x w
        preds = preds.view(n_images, ensemble_size, *preds.size()[1:])

        # take average preds
        preds = preds.mean(dim=1)

        return torch.argmax(preds, dim=1)


    def test_time_optimize(self, target_image, sample_size=8):
        n_images, n_channels = target_image.size()[0:2]
        ## perform test time optimization for finding best test time aug policies
        ## the style parameters are initialized to have zero mean and zero std (not random).
        ## First style component is gamma = 1 +- s[0], second style component is alpha = 1 +- s[1] and third style component is beta = +- s[2]
        ##
        mu_styles = torch.zeros(3, n_images, n_channels, 1, dtype=torch.float32, device=target_image.device) # last dim for sample size
        mu_styles.requires_grad_(True)
        ##
        std_styles = 1e-3 + torch.zeros(3, n_images, n_channels, 1, dtype=torch.float32, device=target_image.device)
        std_styles.requires_grad_(True)

        ## optimizer
        optimizer = torch.optim.Adam([{'params': [mu_styles, std_styles]}], lr=0.01, betas=(0.9,0.999), eps=1e-8)
        
        schedular = self.get_schedular(optimizer, linear_steps=0, const_setps=self.opt.n_steps, cosine_steps=0)


        ## visualizations
        series_visuals = []
        series_visuals_imgs = []
        series_losses = []

        # Pre-visual
        seg_pred = self.unet(target_image)
        series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True).detach().cpu())
        series_visuals_imgs.append(target_image.detach().cpu())

        target_image = target_image.unsqueeze(2) # shape is now n_images x n_channels x 1 x h x w

        ## we update the style code to fix the segmentation output of UNet
        for i in tqdm.tqdm(range(self.opt.n_steps)):
            optimizer.zero_grad()
            # sample styles from the distribution. mu_style : 3 x n_images x n_channels
            iter_style = mu_styles + torch.abs(std_styles) * torch.randn(3, n_images, n_channels, sample_size, dtype=torch.float32, device=target_image.device)
            # shape of iter_style is 3, n_images, n_channels, sample_size: see above why new dim is added to target_image

            ## reconstrut image using style manipulator
            syn_img = self.style_manipulator(target_image, iter_style) # n_images x n_channels x sample_size x h x w
            syn_img = syn_img.permute(0, 2, 1, 3, 4).flatten(0, 1) # n_images x sample_size x n_channels x h x w -> (n_images.sample_size) x n_channels x h x w

            seg_pred, feats = self.unet(syn_img, feats=True)

            ## smoothness loss
            losses = self.smooth_loss(seg_pred, feats, mu_styles, std_styles)
            loss = sum([v for v in losses.values()])
            loss.backward()

            optimizer.step()
            schedular.step()

            if i % 10 == 0:
                series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True).detach().cpu())
                series_visuals_imgs.append(syn_img.detach().cpu())

            # for visulaization
            series_losses.append([v.detach().cpu().item() for v in losses.values()] + [loss.detach().cpu().item(), mu_styles.abs().detach().mean().cpu().item(), std_styles.abs().detach().mean().cpu().item()])
        
        ## final predictions
        target_image = target_image.squeeze(2)

        if self.opt.n_steps > 0:
            ## final predictions
            target_image = target_image.squeeze(2)

            final_preds = self.get_ensemble_predictions(mu_styles.detach(), std_styles.detach(), target_image, 16).detach().cpu()
        else:
            final_preds = torch.argmax(seg_pred, dim=1)
        

        ## process the data
        series_losses = torch.tensor(series_losses)
        series_visuals = tvu.make_grid(torch.cat(series_visuals, dim=0), nrow=4)[0] # make_grid makes the channel size to 3
        series_visuals_imgs = tvu.make_grid(torch.cat(series_visuals_imgs, dim=0), nrow=4)
        series_visuals = overlay_segs(series_visuals_imgs, series_visuals)


        return final_preds, series_losses, series_visuals, series_visuals_imgs

    def save_plots(self, x, img_name, legend):
        x = x.detach().cpu().numpy()

        for i in range(x.shape[1]):
            plt.plot(np.arange(x.shape[0]), x[:, i])

        plt.legend(legend)
        plt.grid()
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
                    self.save_plots(losses, all_imgs[i], ['entropy', 'batchnorm_consistency', 'style_constraints_mu', 'style_constraints_std', 'total', 'mu', 'std'])
                
                self.save_visuals(visuals, 'segmentations_iters_' + all_imgs[i])
                self.save_visuals([img[i:(i + BATCH)].repeat([1, 3, 1, 1]).clamp(-1, 1) * 0.5 + 0.5, overlay_segs(img[i:(i + BATCH)], seg[i:(i + BATCH)]), overlay_segs(img[i:(i + BATCH)], pred)],
                                'predictions_' + all_imgs[i])

                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0)

            for i in range(len(all_segs)):
                self.save_pred_numpy(predictions[i], all_segs[i])                 


