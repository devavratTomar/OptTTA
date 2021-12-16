from kornia.losses import dice
import torch
import os
import losses
from networks.resnet import resnet34

from data import SkinDataset
from torch.utils.data import DataLoader
from losses import CrossEntropyLossWeighted

import kornia as K
import numpy as np

from util import IterationCounter, Visualizer, MetricTracker, compute_accuracy, print_text_on_images
from tqdm import tqdm


class SourceDomainTrainerClassification():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        ### initialize dataloaders
        self.train_dataloader = DataLoader(
            SkinDataset(self.opt.dataroot, self.opt.source_sites, phase='train', split_train=True, seed=0),
            batch_size=self.opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.opt.n_dataloader_workers
        )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            SkinDataset(self.opt.dataroot, self.opt.source_sites, phase='val', split_train=True, seed=0),
            batch_size=self.opt.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models
        self.model = resnet34(num_classes=self.opt.n_classes)

        # in train mode
        self.model.train()

        ## load models if needed
        if self.opt.continue_train:
            self.load_models(self.opt.resume_iter)

        ## use gpu
        if self.opt.use_gpu:
            self.model = self.model.to(self.opt.gpu_id)


        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)


        ## losses weight=torch.tensor(self.train_dataloader.dataset.weights, dtype=torch.float32, device=self.opt.gpu_id),
        self.criterian_wce = losses.FocalLoss(gamma=2) #torch.nn.CrossEntropyLoss(reduction='mean')
        # self.criterian_focal = 

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.metric_tracker = MetricTracker()
    
    def load_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Classifier_%s.pth'%epoch), map_location='cpu')
        self.model.load_state_dict(weights)

    def save_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir
        torch.save(self.model.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Classifier_%s.pth'%epoch))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3) # maximize dice score
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
        labels = data[1]

        predict = self.model(imgs)

        loss_ce = self.criterian_wce(predict, labels)

        self.grad_scaler.scale(loss_ce).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        seg_losses = {}
        seg_losses['train_ce'] = loss_ce.detach()

        return seg_losses
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        labels = data[1]

        losses = {}
        predict = self.model(imgs)
        losses['val_wce'] = self.criterian_wce(predict, labels).detach()

        self.model.train()

        return losses

    @torch.no_grad()
    def compute_metrics_one_step(self, data):
        self.model.eval()
        preds = self.model(data[0]).argmax(dim=1)
        
        # accuracy = compute_accuracy(preds, data[1].detach().cpu())
        accuracy = (preds == data[1]).to(torch.float32).detach().cpu()

        self.model.train()
        return {'accuracy': accuracy}

    @torch.no_grad()
    def get_visuals_for_snapshot(self, data):
        self.model.eval()
        
        # keep display to four
        data[0] = data[0][:4]
        data[1] = data[1][:4]

        gts = data[1]

        imgs = data[0]
        predicts = self.model(imgs).detach().softmax(dim=1).cpu()
        texts = []

        for pred, gt in zip(predicts, gts):
            texts.append('\nPreds:\n' + '\n'.join([self.train_dataloader.dataset.classes_abv[i] \
                + ': %.2f'%(p.item()) for i,p in enumerate(pred)]) + '\nGround Truth: ' + self.train_dataloader.dataset.classes_abv[gt.detach().cpu().item()])

        

        imgs = print_text_on_images(imgs, texts)

        self.model.train()
        return {'imgs_labels': imgs}
    
    def launch(self):
        self.initialize()
        self.train()        

    def train(self):
        train_iterator = iter(self.train_dataloader)
        
        while not self.iter_counter.completed_training():
            with self.iter_counter.time_measurement("data"):
                try:
                    images, labels = next(train_iterator)
                except:
                    train_iterator = iter(self.train_dataloader)
                    images, labels = next(train_iterator)
                
                if self.opt.use_gpu:
                    images = images.to(self.opt.gpu_id)
                    labels = labels.to(self.opt.gpu_id)

            with self.iter_counter.time_measurement("train"):
                losses = self.train_one_step([images, labels])
                self.metric_tracker.update_metrics(losses, smoothe=True)

            with self.iter_counter.time_measurement("maintenance"):
                if self.iter_counter.needs_printing():
                    self.visualizer.print_current_losses(self.iter_counter.steps_so_far,
                                                    self.iter_counter.time_measurements,
                                                    self.metric_tracker.current_metrics())

                if self.iter_counter.needs_displaying():
                    visuals = self.get_visuals_for_snapshot([images, labels])
                    self.visualizer.display_current_results(visuals, self.iter_counter.steps_so_far)
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, self.metric_tracker.current_metrics())
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
                    
                    self.schedular.step(val_losses['val_wce'])
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, val_losses)
                    self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far, val_metrics)

                if self.iter_counter.completed_training():
                    break

                self.iter_counter.record_one_iteration()