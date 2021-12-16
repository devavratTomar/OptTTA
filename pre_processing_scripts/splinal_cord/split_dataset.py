import os
import shutil

def chose_randomly_from_train(rootdir, outdir):
    # sample dataset from the given sites
    all_imgs = sorted([f for f in os.listdir(rootdir) if 'image' in f])
    all_segs = sorted([f for f in os.listdir(rootdir) if 'mask' in f])