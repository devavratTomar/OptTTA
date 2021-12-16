import argparse

def get_source_segmentor_options(parser):
    ## Experiment Specific
    parser.add_argument('--checkpoints_dir', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--resume_iter', default='latest')
    parser.add_argument('--continue_train', action='store_true')

    ## Sizes
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--ncolor_channels', default=1, type=int)
    parser.add_argument('--n_classes', type=int)

    ## Datasets
    parser.add_argument('--dataroot',  type=str) # default='../datasets_slices/spinal_cord/train',
    parser.add_argument('--source_sites', type=str) # default='1,2,3'
    parser.add_argument('--dataset_mode', choices=['spinalcord', 'heart', 'prostate', 'skinlesion']) # default='spinalcord'
    parser.add_argument('--n_dataloader_workers', default=4)
    parser.add_argument('--data_ratio', type=float, default=1.)
    parser.add_argument('--target_UB', action='store_true', help="whether we train on upper bound. We take 3/10 of the data for training")


    ## optimizer
    parser.add_argument("--lr", default=0.00001, type=float)

    ## display
    parser.add_argument("--total_nimgs", default=251000, type=int)
    parser.add_argument("--save_freq", default=50000, type=int)
    parser.add_argument("--evaluation_freq", default=10000, type=int)
    parser.add_argument("--print_freq", default=480, type=int)
    parser.add_argument("--display_freq", default=10000, type=int)
    parser.add_argument("--save_visuals", default=True, type=bool)

    ## test mode
    parser.add_argument("--test", action="store_true", help="whether we enter in test mode")


    opt = parser.parse_args()
    opt.gpu_id = 'cuda:%s'%opt.gpu_id
    if opt.dataset_mode == 'prostate':
        opt.source_sites = ['site-'+ site_nbr for site_nbr in opt.source_sites.split(',')]
    elif opt.dataset_mode == 'skinlesion':
        opt.source_sites = opt.source_sites.split(',')
    else:
        opt.source_sites = ['site'+ site_nbr for site_nbr in opt.source_sites.split(',')]

    if opt.test:
        opt.continue_train = True
        
    return opt