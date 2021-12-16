import torch
import torch.nn.functional as tf
import os
from data import GenericDataset, GenericVolumeDataset
from torch.utils.data import DataLoader

import networks
from networks.unet import UNet

from util.util import overlay_segs
import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm

from PIL import Image

class SourceFreeDomainAdaptor():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):
        #### Test Target Dataloader ####
        self.target_test_dataloader = GenericVolumeDataset(self.opt.dataroot, self.opt.target_sites, self.opt.dataset_mode, phase='test')
            
        ##### load pre-trained style manipulator and unet-segmentor
        self.unet = UNet(self.opt.ncolor_channels, self.opt.n_classes)
        self.encoder_style_manipulator = networks.get_encoder(self.opt)
        self.generator_style_manipulator = networks.get_generator(self.opt)

        self.load_pretrained()

        if self.opt.use_gpu:
            self.unet = self.unet.to(self.opt.gpu_id)
            self.encoder_style_manipulator = self.encoder_style_manipulator.to(self.opt.gpu_id)
            self.generator_style_manipulator = self.generator_style_manipulator.to(self.opt.gpu_id)

        # freeze weights
        self.freeze_weigths(self.unet)
        self.freeze_weigths(self.encoder_style_manipulator)
        self.freeze_weigths(self.generator_style_manipulator)

        # eval mode
        self.unet.eval()
        self.encoder_style_manipulator.eval()
        self.generator_style_manipulator.eval()

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

        # style encoder
        weights = torch.load(os.path.join(self.opt.checkpoints_style, 'saved_models', 'Encoder_850000.pth'), map_location='cpu')
        self.encoder_style_manipulator.load_state_dict(weights)

        # style generator
        weights = torch.load(os.path.join(self.opt.checkpoints_style, 'saved_models', 'Decoder_850000.pth'), map_location='cpu')
        self.generator_style_manipulator.load_state_dict(weights)


    # For style gan based generator
    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.encoder_style_manipulator(sample_image)
            self.generator_style_manipulator(sp, gl)
        noise_var = self.generator_style_manipulator.fix_and_gather_noise_parameters()
        return noise_var
    
    def reset_noise(self):
        self.generator_style_manipulator.remove_noise_parameters()
    

    def get_schedular(self, optim, linear_steps=0, const_setps=100, cosine_steps=0):
        def lr_multiplier(step):
            if step < linear_steps:
                mult = step/linear_steps
            elif linear_steps <= step < linear_steps + const_setps:
                mult = 1.0
            else:
                mult = 0.5*(1.0 + math.cos(math.pi*(step - linear_steps - const_setps)/cosine_steps))

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

    def smooth_loss(self, x, feats, img):
        loss = {}

        p = torch.softmax(x, dim=1)

        # entropy maps
        entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).mean() # E[p log p]
        loss['entropy'] = entropy

        # avg_p = p.mean(dim=[2, 3]) # avg along the pixels dim h x w :=: batch x n_classes
        # init_pred = tf.one_hot(torch.argmax(init_pred, dim=1).flatten(), self.opt.n_classes).to(torch.float32).mean().unsqueeze(0) # batch x n_classes

        #divergence = torch.kl_div(torch.log(avg_p + 1e-6), init_pred).sum(dim=1)
        #loss['divergence'] = 0.05*divergence
        
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

        # loss['mumford_shah'] = 0.001*self.criterian_mumford_shah(img, p)

        # uniformity of probability gent
        # avg_p = p.mean(dim=[0, 2, 3]) # avg along the pixels dim h x w and batch
        # divergence = torch.sum(avg_p * torch.log(avg_p + 1e-6))
        # loss['divergence'] = 0.0001*divergence


        return loss
    

    def test_time_optimize(self, target_image):
        ## perform test time optimization
        ## First fix noise
        self.reset_noise()
        self.fix_noise(target_image)

        ## First we guess the initial avg style and content code using style manipulator
        tgt_content, style = self.encoder_style_manipulator(target_image)
        tgt_content = tgt_content.detach() # fix the content block

        # manipulate style to minimize the smoothness loss on Unet
        iter_style = style.clone().detach().requires_grad_(True)

        ## optimizer
        optimizer = torch.optim.Adam([{'params': [iter_style]}], lr=0.1,betas=(0.9,0.999), eps=1e-8)
        
        schedular = self.get_schedular(optimizer)

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
            syn_img = self.generator_style_manipulator(tgt_content, iter_style)
            seg_pred, feats = self.unet(syn_img, feats=True)

            # if i == 0:
            #     init_predict = seg_pred.detach()

            if i % 5 == 0:
                series_visuals.append(torch.argmax(seg_pred, dim=1, keepdim=True))
                series_visuals_imgs.append(syn_img.detach())

            ## smoothness loss
            losses = self.smooth_loss(seg_pred, feats, target_image)
            loss = sum([v for v in losses.values()])
            loss.backward()
            optimizer.step()
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
                #if torch.max(img) == -1:
                #    pred = torch.zeros_like(pred)
                if self.opt.n_steps > 0:
                    self.save_plots(losses, all_imgs[i], ['entropy', 'batchnorm_consistency', 'total'])
                
                self.save_visuals(visuals, 'segmentations_iters_' + all_imgs[i])
                self.save_visuals([img[i:(i + BATCH)].repeat([1, 3, 1, 1]).clamp(-1, 1) * 0.5 + 0.5, overlay_segs(img[i:(i + BATCH)], seg[i:(i + BATCH)]), overlay_segs(img[i:(i + BATCH)], pred)],
                                'predictions_' + all_imgs[i])

                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0)

            for i in range(len(all_segs)):
                self.save_pred_numpy(predictions[i], all_segs[i])                 


