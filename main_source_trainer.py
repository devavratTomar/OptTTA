import argparse
import os
from trainers import SourceDomainTrainer, SourceDomainTrainerClassification

from options import get_source_segmentor_options

#CUDA_VISIBLE_DEVICES

def ensure_dirs(checkpoints_dir):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
   
    if not os.path.exists(os.path.join(checkpoints_dir,'console_logs')):
        os.makedirs(os.path.join(checkpoints_dir,'console_logs'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'tf_logs')):
        os.makedirs(os.path.join(checkpoints_dir, 'tf_logs'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'saved_models')):
        os.makedirs(os.path.join(checkpoints_dir, 'saved_models'))

    if not os.path.exists(os.path.join(checkpoints_dir, 'visuals')):
        os.makedirs(os.path.join(checkpoints_dir, 'visuals'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Segmentor on Source Images')
    opt_style = get_source_segmentor_options(parser)
    ensure_dirs(opt_style.checkpoints_dir)
    if opt_style.dataset_mode == 'skinlesion':
        trainer = SourceDomainTrainerClassification(opt_style)
    else:
        trainer = SourceDomainTrainer(opt_style)
    trainer.launch()