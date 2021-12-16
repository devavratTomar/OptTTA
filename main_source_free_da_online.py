from options import get_online_adaptation_options
import os

from trainer_policy.ttda_online import TTDAOnline

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Free Adaptation to test time Image.')
    opt_style = get_online_adaptation_options(parser)
    trainer = TTDAOnline(opt_style)
    trainer.launch()