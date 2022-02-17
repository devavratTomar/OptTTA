import torch
from torch._C import device
import torch.nn.functional as tf
import os

from data import GenericDataset, GenericVolumeDataset
from torch.utils.data import DataLoader
import random

import networks
from networks.unet import UNet
from trainer_policy.diff_augmentation import Identity, GaussianBlur, Contrast, Brightness, Gamma, RandomResizeCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotate, DummyAugmentor, RandomScaledCenterCrop

from util.util import overlay_segs, getcolorsegs, clip_gradient
from util.util import natural_sort, ensure_dir
from util import MetricTracker

import torchvision.utils as tvu
import matplotlib.pyplot as plt
import numpy as np
import math
import losses
import tqdm

import itertools

from PIL import Image

DEBUG = True
STYLE_AUGMENTORS = [Gamma.__name__, GaussianBlur.__name__, Contrast.__name__, Brightness.__name__, Identity.__name__]
SPATIAL_AUGMENTORS = [RandomResizeCrop.__name__, RandomHorizontalFlip.__name__, RandomVerticalFlip.__name__, RandomRotate.__name__, RandomScaledCenterCrop.__name__]

STRING_TO_CLASS = {
    'Identity': Identity,
    'GaussianBlur': GaussianBlur,
    'Contrast': Contrast,
    'Brightness': Brightness,
    'Gamma': Gamma,
    'RandomResizeCrop': RandomResizeCrop,
    'RandomHorizontalFlip': RandomHorizontalFlip,
    'RandomVerticalFlip': RandomVerticalFlip,
    'RandomRotate': RandomRotate,
    'RandomScaledCenterCrop': RandomScaledCenterCrop
}

class OptTTA():
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

        self.criterian_entropy = losses.EntropyLoss()
        self.criterian_entropy_cm = losses.EntropyClassMarginals()

        # runnign_vars, means
        self.running_means, self.running_vars = self.get_segmentor_bn_stats()
        self.bn_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] # 18 bn layers

        # metrics tracker
        self.metric_tracker = MetricTracker()
    
    def freeze_weigths(self, net):
        for param in net.parameters():
            param.requires_grad = False

    
    def load_pretrained(self):
        # unet
        latest = natural_sort([f for f in os.listdir(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models')) if f.endswith('.pth')])[-1]
        print('Loading checkpoint: ', latest)
        weights = torch.load(os.path.join(self.opt.checkpoints_source_segmentor, 'saved_models', latest), map_location='cpu')
        self.unet.load_state_dict(weights)


    def save_policy(self, policy_name, augmentors):
        save_dir = os.path.join(self.opt.checkpoints_source_free_da, 'saved_policies', policy_name)
        ensure_dir(save_dir)

        # save augmentors
        for aug in augmentors:
            aug_name = type(aug).__name__
            torch.save(aug.state_dict(), os.path.join(save_dir, aug_name + '.pth'))

    def load_policy(self, policy_name, augmentors):
        save_dir = os.path.join(self.opt.checkpoints_source_free_da, 'saved_policies', policy_name)

        # get augmentors
        for aug in augmentors:
            aug_name = type(aug).__name__
            aug.load_state_dict(torch.load(os.path.join(save_dir, aug_name + '.pth')))

    def load_optimizer(self, policy_name, optimizer):
        save_dir = os.path.join(self.opt.checkpoints_source_free_da, 'saved_policies', policy_name)
        # get optimizer
        optimizer.load_state_dict(torch.load(os.path.join(save_dir, 'optimizer.pth')))

    def save_optimizer(self, policy_name, optimizer_state_dict):
        save_dir = os.path.join(self.opt.checkpoints_source_free_da, 'saved_policies', policy_name)
        ensure_dir(save_dir)

        # save optimizer state dict
        torch.save(optimizer_state_dict, os.path.join(save_dir, 'optimizer.pth'))
    

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
        loss['entropy'] = self.criterian_entropy(p)
        
        # match bn stats
        loss_bn = 0
        for i, (f, m, v) in enumerate(zip(feats, self.running_means, self.running_vars)):
            if i in self.bn_layers:
                # b x ch x h x w
                current_mean = f.mean(dim=(0, 2, 3))
                cuurent_var  = f.var(dim=(0, 2, 3))

                loss_bn += self.criterian_l2(current_mean, m) + self.criterian_l2(cuurent_var, v)

        loss['batchnorm_consistency'] = self.opt.alpha_1*loss_bn
        
        loss['entropy_class_marginal'] = self.opt.alpha_2*self.criterian_entropy_cm(p)
        
        return loss
    
    @torch.no_grad()
    def ensemble_predictions(self, augmentors, tgt_vol, batch_size=16):

        k = self.opt.k
        # self.unet.train()
        style_augmentors = [aug for aug in augmentors if type(aug).__name__ in STYLE_AUGMENTORS]
        spatial_augmentors = [aug for aug in augmentors if type(aug).__name__ in SPATIAL_AUGMENTORS]
        
        predictions_volume = []
        uncertainties_volume = []        
        ###### For visualization ######
        viz_preds = []
        viz_augs = []

        for i in range(tgt_vol.size()[0]):
            predictions = []
            for j in range(0, k//batch_size):
                aug_imgs = tgt_vol[i:(i+1)].repeat(batch_size, 1, 1, 1)

                spatial_affines = []
                for aug in style_augmentors:
                    aug_imgs = aug(aug_imgs).detach()

                for aug in spatial_augmentors:
                    aug_imgs, affine = aug.test(aug_imgs)
                    affine = affine.detach()
                    aug_imgs = aug_imgs.detach()
                    spatial_affines.append(affine)
                
                # get predictions on the augmented images of the ith slice
                preds = self.unet(aug_imgs).detach()

                # visualizations
                if j == 0:
                    viz_preds.append(torch.argmax(preds.cpu(), dim=1))
                    viz_augs.append(aug_imgs.cpu())
                
                # invert affines in reverse order
                for aug, affine in zip(reversed(spatial_augmentors), reversed(spatial_affines)):
                    inv_affine = aug.invert_affine(affine)
                    preds, inv_affine = aug.test(preds, inv_affine)
                    preds = preds.detach()
                    inv_affine = inv_affine.detach()
                
                predictions.append(preds.cpu())

                ## end for j loop
            
            # gather all predictions
            predictions = torch.cat(predictions, dim=0)
            predictions = torch.softmax(predictions, dim=1)

            prediction_labels = torch.argmax(predictions, dim=1, keepdim=False)
            labels_frequency = torch.Tensor()
            for l in range(predictions.shape[1]):
                frequency_l = torch.sum(prediction_labels==l, dim=0, keepdim=True)/k
                labels_frequency = torch.cat([labels_frequency, frequency_l])

            uncertainties = -torch.mean(labels_frequency*torch.log(labels_frequency + 1e-6), dim=0, keepdim=True)
            predictions = torch.mean(predictions, dim=0, keepdim=True)

            predictions_volume.append(predictions)
            uncertainties_volume.append(uncertainties)
            ## end for i loop
        
        # gather predictions of all images of the volume
        predictions_volume = torch.cat(predictions_volume, dim=0)
        uncertainties_volume = torch.cat(uncertainties_volume, dim=0)

        # visualizations
        viz_augs = torch.cat(viz_augs, dim=0)
        viz_preds = torch.cat(viz_preds, dim=0)
        # self.unet.eval()
        return predictions_volume, uncertainties_volume, viz_preds, viz_augs

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

    def optimize_augmentors(self, target_image, augmentors_list, optimizer, n_steps, sample_size):
        ## Augmentors should be on GPU
        print("Optimizing parameters for Augmentations: {}".format(', '.join([type(n).__name__ for n in augmentors_list])))

        if DEBUG:
            pred_visuals = []
            aug_visuals = []
            loss_curve = []

            ## trick if there are a lot of images
            tmp = []
            for j in range(0, target_image.size()[0], sample_size):
                pred = self.unet(target_image[j:(j+sample_size)]).detach().cpu()
                pred = torch.argmax(pred, dim=1, keepdim=True)
                tmp.append(pred)
            
            pred = torch.cat(tmp, dim=0)
            pred_visuals.append(pred)
            aug_visuals.append(target_image.detach().cpu())

        # perform differentiable augmentations
        # training using gradient descent optimization

        # remove all black images from slices
        slice_indices = self.get_slice_index(target_image, 0.2)
        
        for i in tqdm.tqdm(range(n_steps)):
            if optimizer is not None:
                optimizer.zero_grad()

            # apply augmentations
            aug_imgs = target_image[random.choices(slice_indices, k=sample_size)]
            for aug in augmentors_list:
                aug_imgs = aug(aug_imgs)

            pred, feats = self.unet(aug_imgs, feats=True)
            losses = self.smooth_loss(pred, feats)
            loss = sum([v for v in losses.values()])

            if optimizer is not None:
                loss.backward()
                
                clip_gradient(optimizer)

                optimizer.step()

            # free up gpu
            for k in losses.keys():
                losses[k] = losses[k].detach().cpu().item()
            losses['total'] = loss.detach().cpu().item()

            
            # moving average for visualization
            self.metric_tracker.update_metrics(losses)

            pred = pred.detach().cpu()
            loss = loss.detach().cpu().item()
            aug_imgs = aug_imgs.detach().cpu()
            loss_curve.append(list(self.metric_tracker.current_metrics().values()))
            
            # visualizations
            if DEBUG:
                if i % 50 == 0:
                    pred_visuals.append(torch.argmax(pred, dim=1, keepdim=True))
                    aug_visuals.append(aug_imgs)

        if DEBUG:
            return torch.cat(pred_visuals, dim=0), torch.cat(aug_visuals, dim=0), loss_curve

        return loss_curve


    def test_time_optimize(self, target_image, target_image_name, batch_size=12, best_k_policies=3):

        n_augs = self.opt.n_augs
        ## perform test time optimization
        sub_policies = []

        ## check if exploration is done!
        OPT_POLICY_CHECKPOINT = os.path.join(self.opt.checkpoints_source_free_da, 'OptimalSubpolicy.txt')
        if os.path.exists(OPT_POLICY_CHECKPOINT):
            print('\n\nOptimized Sub policies found. Performing exploitation........\n\n')
            with open(OPT_POLICY_CHECKPOINT, 'r') as f:
                OPT_POLICIES = f.readlines()

            for line in OPT_POLICIES:
                subpolicytxt = line.split('_')
                subpolicyclasses = []

                for policy in subpolicytxt:
                    policy = policy.strip()
                    subpolicyclasses.append(STRING_TO_CLASS[policy])

                sub_policies.append(subpolicyclasses)

        else:
            print('\n\nNo Optimized Sub policies exists. Performing exploration........\n\n')

            all_augmentations = [Gamma, GaussianBlur, Contrast, Brightness, Identity, RandomResizeCrop, DummyAugmentor] # DummyAugmentor is combination of random flip and rotate
            for item in itertools.combinations(all_augmentations, n_augs):
                item = list(item)
                if DummyAugmentor in item:
                    item.remove(DummyAugmentor)
                    item += [RandomHorizontalFlip, RandomVerticalFlip, RandomRotate]

                sub_policies.append(item)

        print('\n\n')
        print(['_'.join([v.__name__ for v in sp]) for sp in sub_policies ])
        print('\n\n')

        optimized_subpolicies = []
        subpolicies_optimizers_state_dicts = []
        global_policy_losses = []

        for sub_policy in sub_policies:
            augmentations = []
            policy_name = '_'.join([n.__name__ for n in sub_policy])
            print("Optimizing for sub policies: ", policy_name)

            for policy in sub_policy:
                augmentations.append(policy().to(device=self.opt.gpu_id)) ## create differentiable policies


            ####################################################################################################################
            ### load pre-trained augmentations if needed
            ### check if in exploration phase
            if os.path.exists(OPT_POLICY_CHECKPOINT):
                self.load_policy(policy_name, augmentations)

            ####################################################################################################################
            ### select optimizer
            params = []
            for aug in augmentations:
                params += list(aug.parameters())

            if params:
                optimizer = torch.optim.AdamW(params, lr=self.opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

                ### load state dict if needed
                if os.path.exists(OPT_POLICY_CHECKPOINT):
                    self.load_optimizer(policy_name, optimizer)
            else:
                optimizer = None

            ####################################################################################################################

            ########## number of steps to fine tune 10 times less
            n_steps = self.opt.n_steps//10 if os.path.exists(OPT_POLICY_CHECKPOINT) else self.opt.n_steps

            if not DEBUG:
                loss_curve = self.optimize_augmentors(target_image, augmentations, optimizer, n_steps, batch_size)
            else:
                pred_visuals, aug_visuals, loss_curve = self.optimize_augmentors(target_image, augmentations, optimizer, n_steps, batch_size)

                self.visualize_segmentations(aug_visuals, pred_visuals, policy_name, target_image_name)
                self.visualize_losses(loss_curve, policy_name, target_image_name)

            optimized_subpolicies.append(augmentations)
            subpolicies_optimizers_state_dicts.append(optimizer.state_dict() if optimizer else None)

            if self.opt.sp_selection_metric == "All":
                global_policy_losses.append(loss_curve[-1][-1])
            elif self.opt.sp_selection_metric == "BN":
                global_policy_losses.append(loss_curve[-1][1])
            elif self.opt.sp_selection_metric == "Ent":
                global_policy_losses.append(loss_curve[-1][0])
            else:
                raise NotImplementedError("This selection metric is not defined.")

        best_policy_indices = np.argsort(global_policy_losses)[:best_k_policies]
        all_sub_policy_mean_predictions = {}
        all_sub_policy_uncertainty_estimation = {}
        all_sub_policy_viz_aug = {}
        all_sub_policy_viz_pred = {}

        ## remember what we discovered....
        names_opt_sub_polices = []
        for i in best_policy_indices:
            policy_name = '_'.join([type(n).__name__ for n in optimized_subpolicies[i]])
            print('Loss for policy %s %f'% (policy_name, global_policy_losses[i]))
            names_opt_sub_polices.append(policy_name)

            mean_pred, uncertainty_pred, viz_preds, viz_augs = self.ensemble_predictions(optimized_subpolicies[i], target_image)
            all_sub_policy_mean_predictions[policy_name] = mean_pred
            all_sub_policy_uncertainty_estimation[policy_name] = uncertainty_pred

            ## visualizations
            all_sub_policy_viz_aug[policy_name] = viz_augs
            all_sub_policy_viz_pred[policy_name] = viz_preds

            ## save policies if needed
            self.save_optimizer(policy_name, subpolicies_optimizers_state_dicts[i])
            self.save_policy(policy_name, optimized_subpolicies[i])

        # take average across all subpolicies
        final_prediction = torch.stack(list(all_sub_policy_mean_predictions.values()), dim=0)
        final_prediction = torch.mean(final_prediction, dim=0, keepdim=False)
        final_prediction_labels = torch.argmax(final_prediction, dim=1)

        uncertainty_estimation = torch.stack(list(all_sub_policy_uncertainty_estimation.values()), dim=0)
        uncertainty_estimation = torch.mean(uncertainty_estimation, dim=0, keepdim=False)

        # save opt subpolicy names
        if not os.path.exists(OPT_POLICY_CHECKPOINT):
            with open(OPT_POLICY_CHECKPOINT, 'w') as f:
                for line in names_opt_sub_polices:
                    f.write("%s\n"%line)


        return final_prediction_labels, final_prediction, uncertainty_estimation, all_sub_policy_viz_aug, all_sub_policy_viz_pred


    def visualize_losses(self, x, policy_name, image_name):
        x = np.array(x)
        legend = list(self.metric_tracker.current_metrics().keys())

        for i in range(x.shape[1]):
            plt.plot(np.arange(x.shape[0]), x[:, i])

        plt.legend(legend)
        plt.grid(True)
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'loss_curves', policy_name))
        plt.savefig(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'loss_curves', policy_name, image_name + '.png'))
        plt.clf()

    def visualize_segmentations(self, imgs, segs, policy_name, img_name):
        img_grid = tvu.make_grid(imgs, nrow=4)
        seg_grid = tvu.make_grid(segs, nrow=4)[0] # make_grid makes the channel size to 3
        overlay_grid = overlay_segs(img_grid, seg_grid)

        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'segmentations', policy_name))
        tvu.save_image(overlay_grid, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'segmentations', policy_name, img_name + '.png'))
        tvu.save_image(0.5*img_grid + 0.5, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'segmentations', policy_name, img_name + '_img_' + '.png'))
        tvu.save_image(getcolorsegs(seg_grid), os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'segmentations', policy_name, img_name + '_seg_' + '.png'))

    def visualize_imgs(self, dict_imgs, img_name):
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'images'))

        for k, v in dict_imgs:
            img_grid = tvu.make_grid(0.5*v + 0.5, nrow=4)
            tvu.save_image(img_grid, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'images', k + '_' + img_name + '.png'))


    def save_pred_numpy(self, x, folder, name):
        x = x.detach().cpu().numpy()
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, folder))
        np.save(os.path.join(self.opt.checkpoints_source_free_da, folder, name), x)

    def launch(self):
        self.initialize()
        ensure_dir(os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'final_predictions'))

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
            predictions_prob = []
            uncertainties = []
            BATCH = img.size()[0]
            for i in range(0, img.shape[0], BATCH):
                pred, pred_probs, uncertainty_estimation, all_sub_policy_mean_predictions, all_sub_policy_aug_imgs = self.test_time_optimize(img[i:(i + BATCH)], all_imgs[i])
                viz = [overlay_segs(img[i:(i + BATCH)].detach().cpu(), seg[i:(i + BATCH)].detach().cpu()), overlay_segs(img[i:(i + BATCH)].detach().cpu(), pred.detach().cpu())]
                viz = torch.cat(viz, dim=0)
                viz = tvu.make_grid(viz, nrow=BATCH)
                tvu.save_image(viz, os.path.join(self.opt.checkpoints_source_free_da, 'visuals', 'final_predictions', all_imgs[i] + '.png'))
                uncertainties.append(uncertainty_estimation)
                predictions.append(pred)
                predictions_prob.append(pred_probs)
            
            uncertainties = torch.cat(uncertainties, dim=0)
            predictions = torch.cat(predictions, dim=0)
            predictions_prob = torch.cat(predictions_prob, dim=0)
            print(uncertainties.shape)
            print(predictions.shape)
            print(predictions_prob.shape)

            for i in range(len(all_segs)):
                self.save_pred_numpy(uncertainties[i], 'uncertainties', all_segs[i])
                self.save_pred_numpy(predictions[i], 'predictions', all_segs[i])
                self.save_pred_numpy(predictions_prob[i], 'predictions_prob', all_segs[i])