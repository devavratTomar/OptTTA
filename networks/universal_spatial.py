import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalSpatialManipulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_affine = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).view(1, 2, 3)

    def forward(self, x, affine_code, interpolation='bilinear'):
        ### affine_code is batch [x sample_size] x n_params [number of affine parameters]
        ### x is batch [x sample_size] x n_channels x h x w
        batch_size, sample_size, n_channels, h, w = x.size()

        affine = self.init_affine.to(x.device)
        affine_code = affine + affine_code.view(batch_size*sample_size, 2, 3)

        x = x.view(batch_size*sample_size, n_channels, h, w)

        grid = F.affine_grid(affine_code, x.size())
        x = F.grid_sample(x, grid)

        x = x.view(batch_size, sample_size, n_channels, h, w)

        return x


        
