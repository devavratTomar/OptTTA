import torch
import networks
import os

from data import GenericDataset
from torch.utils.data import DataLoader

class DomainGeneralizationTrainer():
    def __init__(self, opt) -> None:
        self.opt = opt

    def initialize(self):
        #########################################   dataloaders  #########################################
        #### Source Dataloader ####
        self.source_dataloder = DataLoader(
            GenericDataset(self.opt.dataroot, self.opt.source_sites, self.opt.dataset_mode, phase='train_no_color_T'), ## no need to perform style augmentations
            batch_size = self.opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.opt.n_dataloader_workers
        )


        #### Test Target Dataloader ####
        self.target_test_dataloader = DataLoader(
            GenericDataset(self.opt.dataroot, self.opt.target_sites, self.opt.dataset_mode, phase='test'),
            batch_size = self.opt.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.opt.n_dataloader_workers
        )


        #### Style Manipulator.
        self.encoder_style_manipulator = networks.get_encoder(self.opt)
        self.decoder_style_manipulator = networks.get_generator(self.opt)

        #### Load pre-trained weights of style manipulator
        self.load_pretrained()

    def load_pretrained(self):
        weights = torch.load(os.path.join)