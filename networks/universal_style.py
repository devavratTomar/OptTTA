import torch
import torch.nn as nn


class UniversalStyleManipulator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, style_code):
        ### style code is batch x [sample_size] x nchannels x n_styles [or style_dim] 
        ### x is          batch x [sample_size] x nchannels x h x w

        ## compute style params
        gamma = 1.0 + style_code[..., 0] # batch x [sample_size] x nchannels
        alpha = 1.0 + style_code[..., 1] # batch x [sample_size] x nchannels
        beta  =       style_code[..., 2] # batch x [sample_size] x nchannels
        
        # add spatial dims
        gamma.unsqueeze_(-1).unsqueeze(-1)
        alpha.unsqueeze_(-1).unsqueeze(-1)
        beta.unsqueeze_(-1).unsqueeze(-1)

        
        # input image is in the range -1 to 1
        out_img = 0.5*x + 0.5
        out_img = torch.clamp(out_img, 0, 1)

        out_img = out_img**gamma + out_img*alpha + beta

        return 2 * out_img - 1.0