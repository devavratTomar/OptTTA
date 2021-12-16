import argparse

def get_style_augmentor_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_dir', default='./style_checkpoints_universal_coco', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume_iter', default='latest')
    parser.add_argument('--continue_train', action='store_true')

    ## Sizes
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)

    ## Patch Discriminator
    parser.add_argument("--patch_min_scale", default=1 / 8, type=float)
    parser.add_argument("--patch_max_scale", default=1 / 4, type=float)
    parser.add_argument("--patch_num_crops", default=8, type=int)
    parser.add_argument("--patch_use_aggregation", type=bool, default=True)
    parser.add_argument("--netPatchD_scale_capacity", default=4.0, type=float)
    parser.add_argument("--netPatchD_max_nc", default=256 + 128, type=int)
    parser.add_argument("--patch_size", default=128, type=int)
    parser.add_argument("--max_num_tiles", default=8, type=int)

    ## Datasets
    parser.add_argument('--dataroot', default=['../datasets/coco/'])
    parser.add_argument('--targetdataroot', default='../datasets_slices/spinal_cord_no_depth_interpolation/train')
    parser.add_argument('--n_dataloader_workers', default=8)

    ## Networks
    parser.add_argument("--use_antialias", type=bool, default=True)

    parser.add_argument("--spatial_code_ch", default=8, type=int)
    parser.add_argument("--global_code_ch", default=2048, type=int)

    parser.add_argument("--netD_scale_capacity", default=1.0, type=float)
    parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
    parser.add_argument("--netE_num_downsampling_sp", default=4, type=int)
    parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
    parser.add_argument("--netE_nc_steepness", default=2.0, type=float)


    parser.add_argument("--netG_scale_capacity", default=1.0, type=float)
    parser.add_argument("--netG_num_base_resnet_layers", default=2, type=int, help="The number of resnet layers before the upsampling layers.")
    parser.add_argument("--netG_use_noise", type=bool, default=True)
    parser.add_argument("--netG_resnet_ch", type=int, default=256)

    ## Loss Hyperparameters
    parser.add_argument("--lambda_R1", default=10.0, type=float)
    parser.add_argument("--lambda_patch_R1", default=1.0, type=float)
    parser.add_argument("--lambda_L1", default=10.0, type=float)
    parser.add_argument("--lambda_GAN", default=1.0, type=float)
    parser.add_argument("--lambda_PatchGAN", default=1.0, type=float)


    ## optimizer
    parser.add_argument("--lr", default=0.002, type=float)
    parser.add_argument("--beta1", default=0.0, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--R1_once_every", default=16, type=int, help="lazy R1 regularization. R1 loss is computed once in 1/R1_freq times")


    ## display
    parser.add_argument("--total_nimgs", default=25 *(1000 ** 2), type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--evaluation_freq", default=50000, type=int)
    parser.add_argument("--print_freq", default=480, type=int)
    parser.add_argument("--display_freq", default=1600, type=int)
    parser.add_argument("--save_visuals", default=True, type=bool)


    opt = parser.parse_args()

    return opt