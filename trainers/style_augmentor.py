import torch
import networks
import os
from data.dataloaders import GenericDataset, StyleDataset
from torch.utils.data import DataLoader

from losses import gan_loss
import util

from util import IterationCounter
from util import Visualizer
from util import MetricTracker


class StyleAugmentorTrainer():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):
        ### initialize the dataloaders
        self.dataloader = DataLoader(StyleDataset(self.opt.dataroot),
                                     batch_size=self.opt.batch_size,
                                     shuffle=True, drop_last=True, num_workers=self.opt.n_dataloader_workers)

        self.dataloadertarget = DataLoader(GenericDataset(self.opt.targetdataroot, ['site1', 'site2', 'site3', 'site4'],  self.opt.dataset_mode,phase='test'),
                                          batch_size=self.opt.batch_size,
                                          shuffle=True, drop_last=True, num_workers=4)

        #### initialize the models
        self.E = networks.get_encoder(self.opt)
        self.G = networks.get_generator(self.opt)
        self.D = networks.get_discriminator(self.opt)
        self.Dpatch = networks.get_patch_discriminator(self.opt)

        ## load models if needed
        if self.opt.continue_train:
            self.load_models(self.opt.resume_iter)

        # copy models on the gpu
        if self.opt.use_gpu:
            self.E = self.E.cuda()
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.Dpatch = self.Dpatch.cuda()

            ## use multi gpu
            self.E = torch.nn.DataParallel(self.E, device_ids=[0, 1])
            self.G = torch.nn.DataParallel(self.G, device_ids=[0, 1])
            self.D = torch.nn.DataParallel(self.D, device_ids=[0, 1])
            self.Dpatch = torch.nn.DataParallel(self.Dpatch, device_ids=[0, 1])

        # Count the iteration count of the discriminator
        # Used for lazy R1 regularization (c.f. Appendix B of StyleGAN2)
        self.discriminator_iter_counter = 0
        self.train_mode_counter = 0

        # losses
        self.l1_loss = torch.nn.L1Loss()


        # optimizers
        self.optimizer_G, self.optimizer_D = self.get_optimizers()


        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.metric_tracker = MetricTracker(self.opt)



    def get_optimizers(self):
        self.Gparams = list(self.E.parameters()) + list(self.G.parameters())
        optimizer_G = torch.optim.Adam(
            self.Gparams, lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2)
        )

        self.Dparams = list(self.D.parameters()) + list(self.Dpatch.parameters())
        # c.f. StyleGAN2 (https://arxiv.org/abs/1912.04958) Appendix B
        c = self.opt.R1_once_every / (1 + self.opt.R1_once_every)
        optimizer_D = torch.optim.Adam(
            self.Dparams, lr=self.opt.lr * c, betas=(self.opt.beta1 ** c, self.opt.beta2 ** c)
        )

        return optimizer_G, optimizer_D


    def load_models(self, epoch):
        # directory where we save the style checkpoints
        checkpoints_dir = self.opt.checkpoints_dir

        # encoder
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Encoder_%s.pth'%epoch), map_location='cpu')
        self.E.load_state_dict(weights)

        # decoder
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Decoder_%s.pth'%epoch), map_location='cpu')
        self.G.load_state_dict(weights)

        # discriminator
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'Discriminator_%s.pth'%epoch), map_location='cpu')
        self.D.load_state_dict(weights)

        # path discriminator
        weights = torch.load(os.path.join(checkpoints_dir, 'saved_models', 'DiscriminatorPatch_%s.pth'%epoch), map_location='cpu')
        self.Dpatch.load_state_dict(weights)

    def save_models(self, epoch):
        checkpoints_dir = self.opt.checkpoints_dir

        torch.save(self.E.module.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Encoder_%s.pth'%epoch))
        torch.save(self.G.module.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Decoder_%s.pth'%epoch))
        torch.save(self.D.module.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'Discriminator_%s.pth'%epoch))
        torch.save(self.Dpatch.module.state_dict(), os.path.join(checkpoints_dir, 'saved_models', 'DiscriminatorPatch_%s.pth'%epoch))
    
    
    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)
    
    ############################### Swapping AutoEncoder Logic ###############################
    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)
    

    def compute_image_discriminator_losses(self, real, rec, mix):
        # Make sure that the reconstructed image and the mixture image are realistic.
        pred_real = self.D(real)
        pred_rec = self.D(rec)
        pred_mix = self.D(mix)


        losses = {}
        losses["D_real"] = gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN

        losses["D_rec"] = gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        
        losses["D_mix"] = gan_loss(
            pred_mix, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)

        return losses


    def get_random_crops(self, x, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops = util.apply_random_crop(
            x, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops

    def compute_patch_discriminator_losses(self, real, mix):
        losses = {}
        real_feat = self.Dpatch.module.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation
        )
        target_feat = self.Dpatch.module.extract_features(self.get_random_crops(real))
        mix_feat = self.Dpatch.module.extract_features(self.get_random_crops(mix))

        losses["PatchD_real"] = gan_loss(
            self.Dpatch.module.discriminate_features(real_feat, target_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        losses["PatchD_mix"] = gan_loss(
            self.Dpatch.module.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=False,
        ) * self.opt.lambda_PatchGAN

        return losses

    def compute_discriminator_losses(self, real):
        sp, gl = self.E(real)
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."

        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        rec = self.G(sp[:B // 2], gl[:B // 2])
        mix = self.G(self.swap(sp), gl)

        losses = self.compute_image_discriminator_losses(real, rec, mix)
        patch_losses = self.compute_patch_discriminator_losses(real, mix)
        losses.update(patch_losses)

        metrics = {}  # no metrics to report for the Discriminator iteration

        return losses, metrics, sp.detach(), gl.detach()

    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        if self.opt.lambda_patch_R1 > 0.0:
            real_crop = self.get_random_crops(real).detach()
            real_crop.requires_grad_()
            target_crop = self.get_random_crops(real).detach()
            target_crop.requires_grad_()

            real_feat = self.Dpatch.module.extract_features(
                real_crop,
                aggregate=self.opt.patch_use_aggregation)
            target_feat = self.Dpatch.module.extract_features(target_crop)
            pred_real_patch = self.Dpatch.module.discriminate_features(
                real_feat, target_feat
            ).sum()

            grad_real, grad_target = torch.autograd.grad(
                outputs=pred_real_patch,
                inputs=[real_crop, target_crop],
                create_graph=True,
                retain_graph=True,
            )

            dims = list(range(1, grad_real.ndim))
            grad_crop_penalty = grad_real.pow(2).sum(dims) + \
                grad_target.pow(2).sum(dims)
            grad_crop_penalty *= (0.5 * self.opt.lambda_patch_R1 * 0.5)
        else:
            grad_crop_penalty = 0.0

        losses["D_R1"] = grad_penalty + grad_crop_penalty

        return losses

    def compute_generator_losses(self, real):
        losses, metrics = {}, {}
        B = real.size(0)

        sp, gl = self.E(real)
        rec = self.G(sp[:B // 2], gl[:B // 2])  # only on B//2 to save memory
        sp_mix = self.swap(sp)

        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            real = real[B // 2:]
            gl = gl[B // 2:]
            sp_mix = sp_mix[B // 2:]

        mix = self.G(sp_mix, gl)

        # record the error of the reconstructed images for monitoring purposes
        metrics["L1_dist"] = self.l1_loss(rec, real[:B // 2])

        
        losses["G_L1"] = metrics["L1_dist"] * self.opt.lambda_L1

        
        losses["G_GAN_rec"] = gan_loss(
            self.D(rec),
            should_be_classified_as_real=True
        ) * (self.opt.lambda_GAN * 0.5)

        losses["G_GAN_mix"] = gan_loss(
            self.D(mix),
            should_be_classified_as_real=True
        ) * (self.opt.lambda_GAN * 1.0)


        real_feat = self.Dpatch.module.extract_features(
            self.get_random_crops(real),
            aggregate=self.opt.patch_use_aggregation).detach()
        mix_feat = self.Dpatch.module.extract_features(self.get_random_crops(mix))

        losses["G_mix"] = gan_loss(
            self.Dpatch.module.discriminate_features(real_feat, mix_feat),
            should_be_classified_as_real=True,
        ) * self.opt.lambda_PatchGAN

        return losses, metrics

    @torch.no_grad()
    def get_visuals_for_snapshot(self, real, prefix=''):
        # avoid the overhead of generating too many visuals during training
        real = real[:4]
        sp, gl = self.E(real)
        sp = sp.detach()
        gl = gl.detach()

        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl).detach()
        mix = self.G(sp, self.swap(gl)).detach()
        visuals = {prefix + "real": real, prefix + "layout": layout, prefix + "rec": rec, prefix + "mix": mix}

        return visuals

    ######################### Training Logic ###############################3
    def train_generator_one_step(self, images):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)
        
        self.optimizer_G.zero_grad()
        g_losses, g_metrics = self.compute_generator_losses(images)
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer_G.step()
        g_losses.update(g_metrics)
        return g_losses

    
    def train_discriminator_one_step(self, images):
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        
        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        
        d_losses, d_metrics, sp, gl = self.compute_discriminator_losses(images)
        self.previous_sp = sp.detach()
        self.previous_gl = gl.detach()
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()

        needs_R1 = self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.opt.R1_once_every == 0
        if needs_R1_at_current_iter:
            self.optimizer_D.zero_grad()
            r1_losses = self.compute_R1_loss(images)
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])
        d_losses.update(d_metrics)
        return d_losses

    def toggle_training_mode(self):
        modes = ["discriminator", "generator"]
        self.train_mode_counter = (self.train_mode_counter + 1) % len(modes)
        return modes[self.train_mode_counter]
    
    def train_one_step(self, images):
        images_minibatch = images
        if self.toggle_training_mode() == "generator":
            losses = self.train_discriminator_one_step(images_minibatch)
        else:
            losses = self.train_generator_one_step(images_minibatch)
        return util.to_numpy(losses)

    ## main function to launch the trainnig process
    def launch(self):
        # initialize
        self.initialize()
        iterator = iter(self.dataloader)
        iterator_target = iter(self.dataloadertarget)

        while not self.iter_counter.completed_training():
            with self.iter_counter.time_measurement("data"):
                try:
                    images = next(iterator)
                except:
                    iterator = iter(self.dataloader)
                    images = next(iterator)
                
                if self.opt.use_gpu:
                    images = images.cuda()
            
            with self.iter_counter.time_measurement("train"):
                losses = self.train_one_step(images)
                self.metric_tracker.update_metrics(losses, smoothe=True)

            with self.iter_counter.time_measurement("maintenance"):
                if self.iter_counter.needs_printing():
                    self.visualizer.print_current_losses(self.iter_counter.steps_so_far,
                                                    self.iter_counter.time_measurements,
                                                    self.metric_tracker.current_metrics())
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, self.metric_tracker.current_metrics())

                if self.iter_counter.needs_displaying():
                    visuals = self.get_visuals_for_snapshot(images, 'orgdata')
                    self.visualizer.display_current_results(visuals,
                                                    self.iter_counter.steps_so_far)
                    try:
                        images, _ = next(iterator_target)
                    except:
                        iterator_target = iter(self.dataloadertarget)
                        images, _ = next(iterator_target)
                    
                    if self.opt.use_gpu:
                        images = images.cuda()
                    
                    visuals = self.get_visuals_for_snapshot(images, 'tgtdata')
                    self.visualizer.display_current_results(visuals,
                                                    self.iter_counter.steps_so_far)             

                if self.iter_counter.needs_saving():
                    self.save_models(self.iter_counter.steps_so_far)

                if self.iter_counter.completed_training():
                    break

                self.iter_counter.record_one_iteration()

            

