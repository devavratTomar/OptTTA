from networks.models_style import StyleGAN2ResnetGenerator
from networks.models_style import StyleGAN2Discriminator
from networks.models_style import StyleGAN2PatchDiscriminator
from networks.models_style import StyleGAN2ResnetEncoder
from networks.bias_correction import BiasCorrection
from networks.universal_style import UniversalStyleManipulator
from networks.universal_spatial import UniversalSpatialManipulator


def get_encoder(opt):
    return StyleGAN2ResnetEncoder(opt)

def get_patch_discriminator(opt):
    return StyleGAN2PatchDiscriminator(opt)

def get_discriminator(opt):
    return StyleGAN2Discriminator(opt)

def get_generator(opt):
    return StyleGAN2ResnetGenerator(opt)
