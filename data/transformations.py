import albumentations as A
from albumentations.augmentations import transforms


def get_transform(phase):
    if phase == 'train':
        return [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(p=0.5),
                A.GaussianBlur(),
                A.RandomBrightness(0.2),
                A.RandomContrast(0.2),
                A.RandomGamma(),
                A.RandomResizedCrop(256, 256, scale=(0.5, 1.0)),
            ]
        
    elif phase == 'train_no_color_T':
        return [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(p=0.5),
                A.RandomResizedCrop(256, 256, scale=(0.5, 1.0)),
            ]

    elif phase == 'test' or phase == 'val' or phase == 'train_no_T':
        return [A.Resize(256, 256)]
