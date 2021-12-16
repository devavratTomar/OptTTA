import SimpleITK as sitk
import os
import numpy as np
from PIL import Image
import argparse

def normalize_img(img):
    return (img - img.min())/(img.max() - img.min())

def crop_roi(image):
    size = image.GetSize()
    spacing = image.GetSpacing()

    real_common_crop_size = 250 # site 1 actual size: image size x spacing

    real_img_x = (size[0]*spacing[0] - real_common_crop_size)/2.0
    real_img_y = (size[1]*spacing[1] - real_common_crop_size)/2.0

    if real_img_x < 0:
        real_img_x = 0
    
    if real_img_y < 0:
        real_img_y = 0

    img_x1 = int(real_img_x/spacing[0])
    img_y1 = int(real_img_y/spacing[1])

    crop_size = int(real_common_crop_size/spacing[0])

    img_x2 = img_x1 + crop_size
    img_y2 = img_y1 + crop_size

    if img_x2 > size[0] - 1:
        img_x2 = size[0] - 1
    
    if img_y2 > size[1] - 1:
        img_y2 = size[1] - 1 


    return image[img_x1:img_x2, img_y1: img_y2, :, :]

    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    # return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])


def process_heart_imgs(rootdir, outdir, debug=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if debug and not os.path.exists(os.path.join(outdir, 'debug')):
        os.makedirs(os.path.join(outdir, 'debug'))

    img_paths = sorted([f for f in os.listdir(rootdir) if f.endswith('.nii.gz') and '_gt' not in f])
    seg_paths = sorted([f for f in os.listdir(rootdir) if f.endswith('.nii.gz') and '_gt' in f])

    assert len(img_paths) == len(seg_paths)

    for i in range(len(img_paths)):
        print(img_paths[i])

        cropped_img = sitk.ReadImage(os.path.join(rootdir, img_paths[i]))
        cropped_seg = sitk.ReadImage(os.path.join(rootdir, seg_paths[i]))

        cropped_img = crop_roi(cropped_img)
        cropped_seg = crop_roi(cropped_seg)

        print(cropped_img.GetSize())

        # extract array from the cropped img and seg.
        # note that not all the time stamps are labelled. img shape is ts x depth x w x h
        img_array = sitk.GetArrayFromImage(cropped_img)
        seg_array = sitk.GetArrayFromImage(cropped_seg)

        labelled_ts = np.unique(np.where(seg_array != 0)[0])
        
        print(labelled_ts)

        assert img_array.shape == seg_array.shape

        for ts in labelled_ts:
            for j in range(img_array.shape[1]):
                img_slice = normalize_img(img_array[ts,j])
                seg_slice = seg_array[ts, j]
                if seg_slice.sum() == 0: # remove slices without any labels
                    continue

                np.save(os.path.join(outdir, str(ts) + '-' + img_paths[i] + '-' + str(j)), img_slice)
                np.save(os.path.join(outdir, str(ts) + '-' + seg_paths[i] + '-' + str(j)), seg_slice)

                if debug:
                    debug_img_slice = (255*img_slice).astype(np.uint8)
                    debug_seg_slice = (50*seg_slice).astype(np.uint8)

                    Image.fromarray(debug_img_slice).save(os.path.join(outdir, 'debug', str(ts) + '-' + img_paths[i] + '-' + str(j) + '.png'))
                    Image.fromarray(debug_seg_slice).save(os.path.join(outdir, 'debug', str(ts) + '-' + seg_paths[i] + '-' + str(j) + '.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Nii Volumes for Spinal Cord Dataset')

    parser.add_argument('--rootdir', help='The root directory for the input images.')
    parser.add_argument('--outdir', help='Output directory for the processed images.')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    process_heart_imgs(args.rootdir, args.outdir, args.debug)
