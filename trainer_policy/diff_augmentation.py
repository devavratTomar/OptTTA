import torch
import torch.nn as nn
import torch.nn.functional as F
import random

############################################################ Modules to change the appearance of the input images ############################################################
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class CommonStyle(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1, dtype=torch.float32), requires_grad=True) # batch x ch x h x w
        self.range = nn.Parameter(1e-3*torch.ones(1, n_channels, 1, 1 , dtype=torch.float32), requires_grad=True)

        self.uniform_dist = lambda size, device : 2*torch.rand(size=size, device=device) - 1 # uniform distribution in the range [-1, 1]

    def denormalize(self, x):
        ## input image is in the range -1, to 1
        #  return torch.clamp(0.5*x + 0.5, 0, 1)
        return 0.5*x + 0.5
    
    def normalize(self, x):
        ## input image is in the range 0 to 1
        return 2 * x - 1

    def forward(self, x):
        raise NotImplementedError

#### blurring transform
class GaussianBlur(nn.Module):
    def __init__(self, nchannels=1, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padder = nn.ReflectionPad2d(kernel_size//2)

        ## register base gaussian window as buffer
        self.register_buffer('base_gauss', self.get_gaussian_kernel2d(kernel_size).repeat(nchannels, 1, 1, 1))

        # register controllabel gaussian blur param
        self.register_parameter('sigma', nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True))

    def gaussian_window(self, window_size):
        def gauss_fcn(x):
            return -(x - window_size // 2)**2 / 2.0
        
        gauss = torch.stack(
            [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
        return gauss

    def get_gaussian_kernel(self, ksize):
        window_1d = self.gaussian_window(ksize)
        return window_1d
    
    def get_gaussian_kernel2d(self, ksize):
        kernel_x = self.get_gaussian_kernel(ksize)
        kernel_y = self.get_gaussian_kernel(ksize)
        kernel_2d = torch.matmul(
            kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
        return kernel_2d

    def forward(self, x):
        gauss_kernel = self.base_gauss**(1/(self.sigma**2))
        gauss_kernel = gauss_kernel/gauss_kernel.sum()

        x = self.padder(x)
        return F.conv2d(x, gauss_kernel)


class Brightness(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)
    
    def forward(self, x):
        x = self.denormalize(x)

        random_brightness = self.bias + torch.abs(self.range) * self.uniform_dist((x.size()[0], self.n_channels, 1, 1), x.device) # shape is batch x ch x 1 x 1
        x = x + random_brightness #x x.mean(dim=(2, 3), keepdim=True) * 
        x = self.normalize(x)

        return x


class Contrast(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def forward(self, x):
        x = self.denormalize(x)

        random_contrast = 1.0 + self.bias + torch.abs(self.range) * self.uniform_dist((x.size()[0], self.n_channels, 1, 1), x.device) # shape is batch x ch x 1 x 1
        x = x * random_contrast

        x = self.normalize(x)

        return x


class Gamma(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def forward(self, x):
        x = self.denormalize(x)

        random_gamma =  1.0 + self.bias + 1e-2*torch.abs(self.range) * self.uniform_dist((x.size()[0], self.n_channels, 1, 1), x.device) # shape is batch x ch x 1 x 1
        x = x ** (random_gamma)

        x = self.normalize(x)

        return x


############################################################ Modules to change the spatial deformation of the input images ############################################################
class RandomSpatial(nn.Module):
    def __init__(self):
        super().__init__()

        ## buffers
        self.register_buffer('unit_affine', torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))

        ## parameters
        self.register_custom_parameters()
        
        ## uniform distribution getter
        self.uniform_dist = lambda size, device : 2*torch.rand(size=size, device=device) - 1 # uniform distribution in the range [-1, 1]

    def register_custom_parameters(self,):
        raise NotImplementedError

    def generate_random_affine(self, batch_size):
        raise NotImplementedError
    
    def forward(self, x):
        random_affine = self.generate_random_affine(x.size()[0])
        
        grid = F.affine_grid(random_affine, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)

        return x

    @torch.no_grad()
    def test(self, x, random_affine=None):
        if random_affine is None:
            random_affine = self.generate_random_affine(x.size()[0])
        
        grid = F.affine_grid(random_affine, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)

        return x, random_affine

    def get_homographic_mat(self, A):
        H = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant", value=0.0)
        H[..., -1, -1] += 1.0

        return H
    
    def invert_affine(self, affine):
        # affine shape should be batch x 2 x 3
        assert affine.dim() == 3
        assert affine.size()[1:] == torch.Size([2, 3])

        homo_affine = self.get_homographic_mat(affine)
        inv_homo_affine = torch.inverse(homo_affine)
        inv_affine = inv_homo_affine[:, :2, :3]

        return inv_affine

class RandomScaledCenterCrop(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_parameter('delta_scale', nn.Parameter(torch.tensor(0.1, dtype=torch.float32)))
        self.register_buffer('scale_matrix', torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))
    
    def generate_random_affine(self, batch_size):
        affine = (1 - torch.abs(self.delta_scale)) * self.scale_matrix
        affine = affine.repeat(batch_size, 1, 1)
        return affine


class RandomResizeCrop(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_parameter('delta_scale_x', nn.Parameter(torch.tensor(0.1, dtype=torch.float32)))
        self.register_parameter('delta_scale_y', nn.Parameter(torch.tensor(0.1, dtype=torch.float32)))

        self.register_buffer('scale_matrix_x', torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('scale_matrix_y', torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_x', torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))
        self.register_buffer('translation_matrix_y', torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).reshape(-1, 2, 3))

    def get_random(self, batch_size, device):
        return 2*torch.rand(batch_size, 1, 1, device=device) - 1.0
    
    def generate_random_affine(self, batch_size):
        delta_x = 0.5 * self.delta_scale_x * self.get_random(batch_size, self.delta_scale_x.device)
        delta_y = 0.5 * self.delta_scale_y * self.get_random(batch_size, self.delta_scale_y.device)

        affine = (1 - torch.abs(self.delta_scale_x)) * self.scale_matrix_x + (1 - torch.abs(self.delta_scale_y)) * self.scale_matrix_y +\
                  delta_x * self.translation_matrix_x + \
                  delta_y * self.translation_matrix_y

        # affine = affine.repeat(batch_size, 1, 1)
        return affine


class RandomHorizontalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self,):
        self.register_buffer('horizontal_flip', torch.tensor([-1, 0, 0, 0, 1, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_random_affine(self, batch_size):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        # randomly flip some of the images in the batch
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = affine[mask] * self.horizontal_flip

        return affine

class RandomVerticalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self,):
        self.register_buffer('vertical_flip', torch.tensor([1, 0, 0, 0, -1, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_random_affine(self, batch_size):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        # randomly flip some of the images in the batch
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = affine[mask] * self.vertical_flip

        return affine

class RandomRotate(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.register_buffer('rotation', torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32).reshape(-1, 2, 3))

    def generate_random_affine(self, batch_size):
        affine = self.unit_affine.repeat(batch_size, 1, 1)
        mask = (torch.rand(batch_size, device=self.unit_affine.device) > 0.5)
        affine[mask] = self.rotation.repeat(mask.sum(), 1, 1)

        return affine

class DummyAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x