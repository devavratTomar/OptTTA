import argparse

def get_domain_generalization_options(parser):
    parser.add_argument('--style_checkpoints_dir', default='./style_checkpoints', type=str)


    opt = parser.parse_args()
    return opt