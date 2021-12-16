from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch
import os
import time
import sys
from .util import ensure_dir

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        # tf summary writer
        self.summary_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, 'tf_logs'))

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, 'console_logs', 'loss_log.txt')
        ensure_dir(os.path.join(opt.checkpoints_dir, 'console_logs'))
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False
    

    def display_current_results(self, visuals, epoch, max_num_images=4):
        """Display current results on Tensorboard;
        save current results to a Folder file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        for img_key, images in visuals.items():
            # images should be ranging from 0 and 1
            images = images[:max_num_images]
            images = images.clamp(-1, 1)*0.5 + 0.5
            self.summary_writer.add_images(img_key, images, epoch)

            if self.opt.save_visuals:
                grid_imgs = vutils.make_grid(images, max_num_images)
                vutils.save_image(grid_imgs, os.path.join(self.opt.checkpoints_dir, 'visuals', img_key + str(epoch) + '.png'))

    def plot_current_losses(self, epoch, losses):
        self.summary_writer.add_scalars('Losses', losses, epoch)

    def plot_current_metrics(self, epoch, metrics):
        self.summary_writer.add_scalars('Metrics', metrics, epoch)

    def plot_current_histogram(self, epoch, data):
        for k, v in data.items():
            self.summary_writer.add_histogram('Histogram/' + k, v, epoch)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, iters, times, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(iters: %d' % (iters)
        for k, v in times.items():
            message += ", %s: %.3f" % (k, v)
        message += ") "
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v.mean())

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
