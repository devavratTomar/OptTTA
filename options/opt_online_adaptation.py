import argparse

def get_online_adaptation_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_source_free_da', type=str)
    parser.add_argument('--checkpoints_source_segmentor', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume_iter', default='latest')

    ## Sizes
    parser.add_argument('--sample_size', default=16, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', type=int)

    ## Datasets
    parser.add_argument('--dataroot',  type=str) # default='../datasets_slices/spinal_cord/train',
    parser.add_argument('--psudo_root', type=str)
    parser.add_argument('--target_sites', type=str) # default='1,2,3'
    parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate']) # default='spinalcord'


    ## optimizer
    parser.add_argument("--lr", default=2e-4, type=float)

    ## display
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--evaluation_freq", default=500, type=int)
    parser.add_argument("--print_freq", default=100, type=int)


    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s'%opt.gpu_id
    if opt.dataset_mode == 'prostate':
        opt.target_sites = ['site-'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    else:
        opt.target_sites = ['site'+ site_nbr for site_nbr in opt.target_sites.split(',')]
    
        
    return opt