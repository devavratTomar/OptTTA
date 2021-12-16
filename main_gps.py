import os
import argparse
from options import get_source_free_domain_adaptaion_options

from randaug.gps import GPS


def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'predictions')):
        os.makedirs(os.path.join(checkpoints_dir, 'predictions'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'uncertainties')):
        os.makedirs(os.path.join(checkpoints_dir, 'uncertainties'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPS')
    opt_style = get_source_free_domain_adaptaion_options(parser)
    ensure_dirs(opt_style.checkpoints_source_free_da)
    trainer = GPS(opt_style)
    trainer.launch()