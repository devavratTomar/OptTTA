from kornia.losses import dice
import torch
import os
import losses
from networks.unet import UNet

from data import GenericDataset
from torch.utils.data import DataLoader
from losses import CrossEntropyLossWeighted

import kornia as K
import numpy as np

from util import IterationCounter, Visualizer, MetricTracker, segmentation_score_stats, dice_coef_multiclass
from util.util import COLORS, overlay_segs
from tqdm import tqdm


class SourceDomainTrainer():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        if self.opt.test:

            self.train_dataloader=None
            self.val_dataloader=None
            self.test_dataloader = DataLoader(
                GenericDataset(self.opt.dataroot, self.opt.source_sites, self.opt.dataset_mode, phase='test', target=self.opt.target_UB),
                batch_size=self.opt.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=4
            )

            print('Length of test dataset: ', len(self.test_dataloader))

        else:
            ### initialize dataloaders
            self.train_dataloader = DataLoader(
                GenericDataset(self.opt.dataroot, self.opt.source_sites, self.opt.dataset_mode, phase='train', split_train=True, seed=0, target=self.opt.target_UB, data_ratio=self.opt.data_ratio),
                batch_size=self.opt.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.opt.n_dataloader_workers
            )

            print('Length of training dataset: ', len(self.train_dataloader))

            self.val_dataloader = DataLoader(
                GenericDataset(self.opt.dataroot, self.opt.source_sites, self.opt.dataset_mode, phase='val', split_train=True, seed=0, target=self.opt.target_UB),
                batch_size=self.opt.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=4
            )

            print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models
        self.model = UNet(self.opt.ncolor_channels, self.opt.n_classes)

        ## load models if needed
        if self.opt.continue_train:
            self.load_models(self.opt.resume_iter)

        ## use gpu
        if self.opt.use_gpu:
            self.model = self.model.to(self.opt.gpu_id)


        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)


        ## losses
        self.criterian_wce = CrossEntropyLossWeighted(self.opt.n_classes)
        self.criterian_dc  = K.losses.DiceLoss()

        ## metrics
        self.dice_coef = dice_coef_multiclass

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.metric_tracker = MetricTracker()

    
    def load_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Segmentor_%s.pth'%epoch), map_location='cpu')
        self.model.load_state_dict(weights)

    def save_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        torch.save(self.model.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Segmentor_%s.pth'%epoch))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.RMSprop(params, lr=self.opt.lr, weight_decay=1e-4, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2) # maximize dice score
        return optimizer, scheduler
    
    @torch.no_grad()
    def get_bn_stats(self):
        running_means = []
        running_vars = []

        for l in self.model.modules():
            if isinstance(l, torch.nn.BatchNorm2d):
                running_means.append(l.running_mean.flatten().detach())
                running_vars.append(l.running_var.flatten().detach())

        running_means = torch.cat(running_means).cpu().numpy()
        running_vars = torch.cat(running_vars).cpu().numpy()

        return {'running_mean': running_means, 'running_vars': running_vars}

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        imgs = data[0]
        segs = data[1]

        predict = self.model(imgs)

        loss_ce = self.criterian_wce(predict, segs)
        loss_dc  = self.criterian_dc(predict, segs)
        
        loss = loss_ce + loss_dc
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        seg_losses = {}
        seg_losses['train_dc'] = loss_ce.detach()
        seg_losses['train_wce'] = loss_dc.detach()
        seg_losses['train_total'] = loss.detach()

        return seg_losses
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        segs = data[1]

        losses = {}
        predict = self.model(imgs)
        losses['val_wce'] = self.criterian_wce(predict, segs).detach()
        losses['val_dc'] = self.criterian_dc(predict, segs).detach()

        self.model.train()

        return losses

    @torch.no_grad()
    def compute_metrics_one_step(self, data):

        self.model.eval()

        imgs = data[0]
        segs = data[1]

        metrics = {}
        predict = self.model(imgs)

        # get classes to evaluate except background
        all_classes = np.arange(self.opt.n_classes)

        # compute dice coefficients
        batch_dice_coef = self.dice_coef(segs.detach().cpu(), 
                                        torch.argmax(predict, dim=1).detach().cpu(), 
                                        all_classes)

        # compute class-wise
        for i, coef in enumerate(batch_dice_coef.T):
            metrics["ds_class_{:d}".format(i)] = torch.tensor(coef)

        # compute sample-wise mean (w/o background)
        metrics["ds"] = torch.tensor(np.nanmean(batch_dice_coef[:,1:], axis=1))

        self.model.train()

        return metrics


    @torch.no_grad()
    def get_visuals_for_snapshot(self, data):
        self.model.eval()
        
        # keep display to four
        data[0] = data[0][:4]
        data[1] = data[1][:4]

        imgs = data[0]
        # segs = COLORS[data[1]].permute(0, 3, 1, 2)
        segs = 2*overlay_segs(imgs, data[1]) - 1
        predicts = self.model(imgs).detach()

        # predicts = COLORS[torch.argmax(predicts, dim=1)].permute(0, 3, 1, 2)
        predicts = 2*overlay_segs(imgs, torch.argmax(predicts, dim=1)) - 1
        self.model.train()
        return {'imgs': imgs, 'segs': segs, 'preds': predicts}
    
    def launch(self):
        self.initialize()

        if self.opt.test:
            self.test()
        else:
            self.train()
        
    @torch.no_grad()
    def test(self):

        test_metrics = None

        for test_it, (test_imgs, test_segs) in enumerate(tqdm(self.test_dataloader)):

            if self.opt.use_gpu:
                test_imgs = test_imgs.to(self.opt.gpu_id)
                test_segs = test_segs.to(self.opt.gpu_id)

            # compute dice coefficients
            if test_metrics is None:
                test_metrics = self.compute_metrics_one_step((test_imgs, test_segs))
            else:
                for k, v in self.compute_metrics_one_step((test_imgs, test_segs)).items():
                    test_metrics[k] = torch.cat((test_metrics[k], v))

        segmentation_score_stats(test_metrics)

        

    def train(self):
        train_iterator = iter(self.train_dataloader)
        
        while not self.iter_counter.completed_training():
            with self.iter_counter.time_measurement("data"):
                try:
                    images, segs = next(train_iterator)
                except:
                    train_iterator = iter(self.train_dataloader)
                    images, segs = next(train_iterator)
                
                if self.opt.use_gpu:
                    images = images.to(self.opt.gpu_id)
                    segs = segs.to(self.opt.gpu_id)

            with self.iter_counter.time_measurement("train"):
                losses = self.train_one_step([images, segs])
                self.metric_tracker.update_metrics(losses, smoothe=True)

            with self.iter_counter.time_measurement("maintenance"):
                if self.iter_counter.needs_printing():
                    self.visualizer.print_current_losses(self.iter_counter.steps_so_far,
                                                    self.iter_counter.time_measurements,
                                                    self.metric_tracker.current_metrics())

                if self.iter_counter.needs_displaying():
                    visuals = self.get_visuals_for_snapshot([images, segs])
                    self.visualizer.display_current_results(visuals, self.iter_counter.steps_so_far)
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, losses)
                    self.visualizer.plot_current_histogram(self.iter_counter.steps_so_far, self.get_bn_stats())

                if self.iter_counter.needs_saving():
                    self.save_models(self.iter_counter.steps_so_far)

                if self.iter_counter.needs_evaluation():
                    val_losses = None
                    val_metrics = None
                    for val_it, (val_imgs, val_segs) in enumerate(self.val_dataloader):
                        if val_it > 100:
                            break
                        if self.opt.use_gpu:
                            val_imgs = val_imgs.to(self.opt.gpu_id)
                            val_segs = val_segs.to(self.opt.gpu_id)

                        if val_losses is None:
                            val_losses = self.validate_one_step([val_imgs, val_segs])
                        else:
                            for k, v in self.validate_one_step([val_imgs, val_segs]).items():
                                val_losses[k] += v

                        if val_metrics is None:
                            val_metrics = self.compute_metrics_one_step([val_imgs, val_segs])
                        else:
                            for k, v in self.compute_metrics_one_step([val_imgs, val_segs]).items():
                                val_metrics[k] = torch.cat((val_metrics[k], v))
                    
                    for k, v in val_losses.items():
                        val_losses[k] = v/(val_it+1)

                    for k, v in val_metrics.items():
                        val_metrics[k] = np.nanmean(v.numpy())
                    
                    self.schedular.step(val_metrics['ds'])
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, val_losses)
                    self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far, val_metrics)

                if self.iter_counter.completed_training():
                    break

                self.iter_counter.record_one_iteration()