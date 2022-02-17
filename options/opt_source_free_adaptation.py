import argparse

def get_source_free_domain_adaptaion_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_source_free_da', type=str)
    parser.add_argument('--checkpoints_source_segmentor', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    ## Sizes
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', default=3, type=int)

    ## Datasets
    parser.add_argument('--dataroot', type=str) # default='../datasets_slices/spinal_cord_no_depth_interpolation/train'
    parser.add_argument('--target_sites')
    parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate'])

    ## Networks
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--alpha_1", default=0.1, type=float)
    parser.add_argument("--alpha_2", default=0.1, type=float)
    parser.add_argument("--n_augs", default=5, type=int)
    parser.add_argument("--k", default=128, type=int)
    parser.add_argument("--sp_selection_metric", default="All", type=str, choices=("All, BN, Ent"))


    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s'%opt.gpu_id
    if opt.dataset_mode == 'prostate':
        opt.target_sites = ['site-'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    else:
        opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    return opt