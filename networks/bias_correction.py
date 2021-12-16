import torch
import torch.nn as nn


class BiasCorrection(nn.Module):
    def __init__(self, input_ch):
        super().__init__()

        self.model = nn.Sequential(nn.Conv2d(input_ch, 64, 5, 1, 2, bias=False),
                                   nn.ReLU(True),
                                   nn.Conv2d(64, 1, 5, 1, 2))


    def forward(self, x):
        return self.model(x)