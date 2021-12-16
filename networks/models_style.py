from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from networks.networks_util.stylegan2_layers import Discriminator as OriginalStyleGAN2Discriminator
from networks.networks_util.stylegan2_layers import ConvLayer, ResBlock, EqualLinear, ToRGB, StyledConv


###################################### StyleGan2 Discriminator ################################################
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stylegan2_D = OriginalStyleGAN2Discriminator(
            opt.crop_size,
            2.0 * opt.netD_scale_capacity,
            blur_kernel=[1, 3, 3, 1] if self.opt.use_antialias else [1],
            nchannels=self.opt.ncolor_channels
        )

    def forward(self, x):
        pred = self.stylegan2_D(x)
        return pred

    def get_features(self, x):
        return self.stylegan2_D.get_features(x)

    def get_pred_from_features(self, feat, label):
        assert label is None
        feat = feat.flatten(1)
        out = self.stylegan2_D.final_linear(feat)
        return out


################################## Patch Discriminator ###############################################
class BasePatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def needs_regularization(self):
        return False

    def extract_features(self, patches):
        raise NotImplementedError()

    def discriminate_features(self, feature1, feature2):
        raise NotImplementedError()

    def apply_random_transformation(self, patches):
        B, ntiles, C, H, W = patches.size()
        patches = patches.view(B * ntiles, C, H, W)
        transformer = util.RandomSpatialTransformer(self.opt, B * ntiles)
        patches = transformer.forward_transform(patches, (self.opt.patch_size, self.opt.patch_size))

        return patches.view(B, ntiles, C, H, W)

    def sample_patches_old(self, img, indices):
        B, C, H, W = img.size()
        s = self.opt.patch_size
        if H % s > 0 or W % s > 0:
            y_offset = torch.randint(H % s, (), device=img.device)
            x_offset = torch.randint(W % s, (), device=img.device)
            img = img[:, :,
                      y_offset:y_offset + s * (H // s),
                      x_offset:x_offset + s * (W // s)]
        img = img.view(B, C, H//s, s, W//s, s)
        ntiles = (H // s) * (W // s)
        tiles = img.permute(0, 2, 4, 1, 3, 5).reshape(B, ntiles, C, s, s)
        if indices is None:
            indices = torch.randperm(ntiles, device=img.device)[:self.opt.max_num_tiles]
            return self.apply_random_transformation(tiles[:, indices]), indices
        else:
            return self.apply_random_transformation(tiles[:, indices])

    def forward(self, real, fake, fake_only=False):
        assert real is not None
        real_patches, patch_ids = self.sample_patches(real, None)
        if fake is None:
            real_patches.requires_grad_()
        real_feat = self.extract_features(real_patches)

        bs = real.size(0)
        if fake is None or not fake_only:
            pred_real = self.discriminate_features(
                real_feat,
                torch.roll(real_feat, 1, 1))
            pred_real = pred_real.view(bs, -1)


        if fake is not None:
            fake_patches = self.sample_patches(fake, patch_ids)
            fake_feat = self.extract_features(fake_patches)
            pred_fake = self.discriminate_features(
                real_feat,
                torch.roll(fake_feat, 1, 1))
            pred_fake = pred_fake.view(bs, -1)

        if fake is None:
            return pred_real, real_patches
        elif fake_only:
            return pred_fake
        else:
            return pred_real, pred_fake
  


class StyleGAN2PatchDiscriminator(BasePatchDiscriminator):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        channel_multiplier = self.opt.netPatchD_scale_capacity
        size = self.opt.patch_size
        channels = {
            4: min(self.opt.netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(self.opt.netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(self.opt.netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))

        in_channel = channels[2 ** log_size]

        blur_kernel = [1, 3, 3, 1] if self.opt.use_antialias else [1]

        convs = [('0', ConvLayer(self.opt.ncolor_channels, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ResBlock(in_channel, self.opt.netPatchD_max_nc * 2, downsample=False)))
        convs.append(('6', ConvLayer(self.opt.netPatchD_max_nc * 2, self.opt.netPatchD_max_nc, 3, pad=0)))

        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 2048, activation='fused_lrelu')
        pairlinear2 = EqualLinear(2048, 2048, activation='fused_lrelu')
        pairlinear3 = EqualLinear(2048, 1024, activation='fused_lrelu')
        pairlinear4 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2, pairlinear3, pairlinear4)

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out



################################################# Encoder   ########################################
class StyleGAN2ResnetEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if self.opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer(self.opt.ncolor_channels, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(self.opt.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer(nchannels, self.opt.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True)
            )
        )

        self.DownToGlobalCode = nn.Sequential()
        for i in range(self.opt.netE_num_downsampling_gl):
            idx_from_beginning = self.opt.netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=0)
            )

        nchannels = self.nc(self.opt.netE_num_downsampling_sp +
                            self.opt.netE_num_downsampling_gl)
        self.add_module(
            "ToGlobalCode",
            nn.Sequential(
                EqualLinear(nchannels, self.opt.global_code_ch)
            )
        )

    def nc(self, idx):
        nc = self.opt.netE_nc_steepness ** (5 + idx)
        nc = nc * self.opt.netE_scale_capacity
        # nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.DownToGlobalCode[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.DownToGlobalCode(midpoint)
        x = x.mean(dim=(2, 3))
        gl = self.ToGlobalCode(x)
        sp = util.normalize(sp)
        gl = util.normalize(gl)
        if extract_features:
            return sp, gl, feature
        else:
            return sp, gl



################################################# Generator ########################################
class UpsamplingBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim,
                 blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True,
                                blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False,
                                use_noise=use_noise)

    def forward(self, x, style):
        return self.conv2(self.conv1(x, style), style)


class ResolutionPreservingResnetBlock(torch.nn.Module):
    def __init__(self, opt, inch, outch, styledim):
        super().__init__()
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=False)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=False, bias=False)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = self.skip(x)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class UpsamplingResnetBlock(torch.nn.Module):
    def __init__(self, inch, outch, styledim, blur_kernel=[1, 3, 3, 1], use_noise=False):
        super().__init__()
        self.inch, self.outch, self.styledim = inch, outch, styledim
        self.conv1 = StyledConv(inch, outch, 3, styledim, upsample=True, blur_kernel=blur_kernel, use_noise=use_noise)
        self.conv2 = StyledConv(outch, outch, 3, styledim, upsample=False, use_noise=use_noise)
        if inch != outch:
            self.skip = ConvLayer(inch, outch, 1, activate=True, bias=True)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x, style):
        skip = F.interpolate(self.skip(x), scale_factor=2, mode='bilinear', align_corners=False)
        res = self.conv2(self.conv1(x, style), style)
        return (skip + res) / math.sqrt(2)


class GeneratorModulation(torch.nn.Module):
    def __init__(self, styledim, outch):
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        if style.ndimension() <= 2:
            return x * (1 * self.scale(style)[:, :, None, None]) + self.bias(style)[:, :, None, None]
        else:
            style = F.interpolate(style, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
            return x * (1 * self.scale(style)) + self.bias(style)


class StyleGAN2ResnetGenerator(nn.Module):
    """ The Generator (decoder) architecture described in Figure 18 of
        Swapping Autoencoder (https://arxiv.org/abs/2007.00653).
        
        At high level, the architecture consists of regular and 
        upsampling residual blocks to transform the structure code into an RGB
        image. The global code is applied at each layer as modulation.
        
        Here's more detailed architecture:
        
        1. SpatialCodeModulation: First of all, modulate the structure code 
        with the global code.
        2. HeadResnetBlock: resnets at the resolution of the structure code,
        which also incorporates modulation from the global code.
        3. UpsamplingResnetBlock: resnets that upsamples by factor of 2 until
        the resolution of the output RGB image, along with the global code
        modulation.
        4. ToRGB: Final layer that transforms the output into 3 channels (RGB).
        
        Each components of the layers borrow heavily from StyleGAN2 code,
        implemented by Seonghyeon Kim.
        https://github.com/rosinality/stylegan2-pytorch
    """
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        num_upsamplings = opt.netE_num_downsampling_sp
        blur_kernel = [1, 3, 3, 1] if opt.use_antialias else [1]

        self.global_code_ch = opt.global_code_ch

        self.add_module(
            "SpatialCodeModulation",
            GeneratorModulation(self.global_code_ch, opt.spatial_code_ch))

        in_channel = opt.spatial_code_ch
        for i in range(opt.netG_num_base_resnet_layers):
            # gradually increase the number of channels
            out_channel = (i + 1) / opt.netG_num_base_resnet_layers * self.nf(0)
            out_channel = max(opt.spatial_code_ch, round(out_channel))
            layer_name = "HeadResnetBlock%d" % i
            new_layer = ResolutionPreservingResnetBlock(
                opt, in_channel, out_channel, self.global_code_ch)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        for j in range(num_upsamplings):
            out_channel = self.nf(j + 1)
            layer_name = "UpsamplingResBlock%d" % (2 ** (4 + j))
            new_layer = UpsamplingResnetBlock(
                in_channel, out_channel, self.global_code_ch,
                blur_kernel, opt.netG_use_noise)
            self.add_module(layer_name, new_layer)
            in_channel = out_channel

        last_layer = ToRGB(out_channel, self.global_code_ch,
                           blur_kernel=blur_kernel, out_channel=self.opt.ncolor_channels)
        self.add_module("ToRGB", last_layer)

    def nf(self, num_up):
        ch = 128 * (2 ** (self.opt.netE_num_downsampling_sp - num_up))
        ch = int(min(512, ch) * self.opt.netG_scale_capacity)
        return ch

    def fix_and_gather_noise_parameters(self):
        params = []
        device = next(self.parameters()).device
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                assert m.image_size is not None, "One forward call should be made to determine size of noise parameters"
                m.fixed_noise = torch.nn.Parameter(torch.randn(m.image_size[0], 1, m.image_size[2], m.image_size[3], device=device))
                params.append(m.fixed_noise)
        return params

    def remove_noise_parameters(self):
        for m in self.modules():
            if type(m).__name__ == "NoiseInjection":
                m.fixed_noise = None
                m.image_size = None

    def forward(self, spatial_code, global_code):
        spatial_code = util.normalize(spatial_code)
        global_code = util.normalize(global_code)

        x = self.SpatialCodeModulation(spatial_code, global_code)
        for i in range(self.opt.netG_num_base_resnet_layers):
            resblock = getattr(self, "HeadResnetBlock%d" % i)
            x = resblock(x, global_code)

        for j in range(self.opt.netE_num_downsampling_sp):
            key_name = 2 ** (4 + j)
            upsampling_layer = getattr(self, "UpsamplingResBlock%d" % key_name)
            x = upsampling_layer(x, global_code)
        rgb = self.ToRGB(x, global_code, None)

        return rgb
