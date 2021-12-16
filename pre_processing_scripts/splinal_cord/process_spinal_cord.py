import os
import argparse

import SimpleITK as sitk

import numpy as np

def normalize_img(img):
    return sitk.RescaleIntensity(img, 0, 1.0)

def crop_roi(image):
    size = image.GetSize()
    spacing = image.GetSpacing()

    real_common_crop_size = 100 * 0.5 # site 1 actual size: image size x spacing

    real_img_x = (size[0]*spacing[0] - real_common_crop_size)/2.0
    real_img_y = (size[1]*spacing[1] - real_common_crop_size)/2.0

    img_x = int(real_img_x/spacing[0])
    img_y = int(real_img_y/spacing[1])
    crop_size = int(real_common_crop_size/spacing[0])

    bounding_box = [img_x, img_y, 0, crop_size, crop_size, size[2]]

    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])


def get_image_names(rootdir):
    image_names = []
    for f in os.listdir(rootdir):
         if f.endswith('nii.gz'):
             split_name = f.split('-')
             f = "-".join(split_name[:2])
             image_names.append(f)
    
    image_names = list(set(image_names))

    return image_names

def set_mask_value(image, mask, value):
    msk32 = sitk.Cast(mask, sitk.sitkFloat32)
    value32 = sitk.Cast(value, sitk.sitkFloat32)

    return sitk.Cast(sitk.Cast(image, sitk.sitkFloat32) * sitk.InvertIntensity(msk32, maximum=1.0) + 
                     msk32*value32, image.GetPixelID())

def merge_masks(d):
    # this does not work as sitk introduce a new label for ties
    voting_filter = sitk.LabelVotingImageFilter()
    mask = voting_filter.Execute(d['mask1'], d['mask2'], d['mask3'], d['mask4'])

    # trusting voter 1 for the ambiguity
    mask = set_mask_value(mask, mask==3, d['mask1'])
    return mask

def process_images(rootdir, outdir, include_seg=False, depth_interpolation=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    image_dicts = []
    file_names = sorted(get_image_names(rootdir))

    for f in file_names:
        if include_seg:
            image_dicts.append({
                'image': sitk.ReadImage(os.path.join(rootdir, f + '-image.nii.gz'), sitk.sitkFloat32),
                'mask1': sitk.ReadImage(os.path.join(rootdir, f + '-mask-r1.nii.gz'), sitk.sitkUInt8),
                'mask2': sitk.ReadImage(os.path.join(rootdir, f + '-mask-r2.nii.gz'), sitk.sitkUInt8),
                'mask3': sitk.ReadImage(os.path.join(rootdir, f + '-mask-r3.nii.gz'), sitk.sitkUInt8),
                'mask4': sitk.ReadImage(os.path.join(rootdir, f + '-mask-r4.nii.gz'), sitk.sitkUInt8)})
        else:
            image_dicts.append({
                'image': sitk.ReadImage(os.path.join(rootdir, f + '-image.nii.gz'), sitk.sitkFloat32)})
    
    if include_seg:
        for img_dict in image_dicts:
            mask = merge_masks(img_dict)
            del img_dict['mask1']
            del img_dict['mask2']
            del img_dict['mask3']
            del img_dict['mask4']

            img_dict['mask'] = mask
        

    modified_data = []

    for img_dict in image_dicts:
        img =  normalize_img(img_dict['image']) # normalize image from 0 to 1

        # crop to the region of interest
        img = crop_roi(img)

        if include_seg:
            seg = crop_roi(img_dict['mask'])
        
            modified_data.append({'image': img, 'mask': seg})
        else:
            modified_data.append({'image': img})

    data = modified_data

    for index, img_dict in enumerate(data):
        print(file_names[index], img_dict['image'].GetSize(), img_dict['image'].GetSpacing())

    for index, img_dict in enumerate(data):

        if depth_interpolation:
            dimension = img_dict['image'].GetDimension()
            # reference_physical_size = np.zeros(dimension)
            reference_physical_size = (np.array(img_dict['image'].GetSize()) - 1)* np.array(img_dict['image'].GetSpacing())

            # we want to make the resolutions same across the sites with spacing 0.25, 0.25 along the x and y diminsion,
            # then center crop the images to 200 x 200

            # for img_dict in data:
            #     reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img_dict['image'].GetSize(),
            #                                                                                         img_dict['image'].GetSpacing(), reference_physical_size)]

            # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
            reference_origin = np.zeros(dimension)
            reference_direction = np.identity(dimension).flatten()

            # reference_size = [100, 100, 32] 
            # reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
            
            # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
            # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
            # spacing set accordingly. 
            # Uncomment the following lines to use this strategy.
            
            reference_size_x = 256
            reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
            reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]


            reference_image = sitk.Image(reference_size, img_dict['image'].GetPixelIDValue())
            reference_image.SetOrigin(reference_origin)
            reference_image.SetSpacing(reference_spacing)
            reference_image.SetDirection(reference_direction)

            if include_seg:
                reference_seg = sitk.Image(reference_size, img_dict['mask'].GetPixelIDValue())
                reference_seg.SetOrigin(reference_origin)
                reference_seg.SetSpacing(reference_spacing)
                reference_seg.SetDirection(reference_direction)

            reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))



            # Transform which maps from the reference_image to the current img with the translation mapping the image
            # origins to each other.
            transform = sitk.AffineTransform(dimension)
            transform.SetMatrix(img_dict['image'].GetDirection())
            transform.SetTranslation(np.array(img_dict['image'].GetOrigin()) - reference_origin)
            
            # Modify the transformation to align the centers of the original and reference image instead of their origins.
            centering_transform = sitk.TranslationTransform(dimension)
            img_center = np.array(img_dict['image'].TransformContinuousIndexToPhysicalPoint(np.array(img_dict['image'].GetSize())/2.0))
            centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
            
            # centered_transform = sitk.Transform(transform)
            # centered_transform.AddTransform(centering_transform)

            # flip:
            # flipped_transform = sitk.AffineTransform(dimension)    
            # flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
            # flipped_transform.SetMatrix([1,0,0,0,-1,0,0,0,1])
            
            centered_transform = sitk.CompositeTransform([transform, centering_transform])


            aug_image = sitk.Resample(img_dict['image'], reference_image, centered_transform, sitk.sitkLinear, 0.0)
            
            if include_seg:
                aug_seg   = sitk.Resample(img_dict['mask'], reference_seg, centered_transform, sitk.sitkNearestNeighbor, 0.0)
            
        else:
            aug_image = img_dict['image']
            if include_seg:
                aug_seg = img_dict['mask']
        
        sitk.WriteImage(aug_image, os.path.join(outdir, file_names[index] + '-image.nii.gz'))
        if include_seg:
            sitk.WriteImage(aug_seg, os.path.join(outdir, file_names[index] + '-mask.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Nii Volumes for Spinal Cord Dataset')

    parser.add_argument('--rootdir', help='The root directory for the input images.')
    parser.add_argument('--outdir', help='Output directory for the processed images.')
    parser.add_argument('--include_seg', action='store_true', help='Include segmentations.')
    parser.add_argument('--no_depth_interpolation', action='store_true', help='Interploate in the depth dimension to make isotropic spacing in depth dimension')
    args = parser.parse_args()
    process_images(args.rootdir, args.outdir, args.include_seg, not args.no_depth_interpolation)
