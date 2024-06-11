import torch
from torchvision import transforms
import monai.transforms
from monai.config import KeysCollection
import torchvision.transforms.functional as TF
from typing import Union
import numpy as np
import random

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# B-scan supervised training Transformations
def bscan_sup_transformsv3(NORM,rotation=10):
    train_transform = transforms.Compose([transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC,antialias=True),
                                      transforms.RandomAffine(0,(0.05,0.05),fill=0, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomRotation(degrees=rotation, interpolation=transforms.InterpolationMode.BILINEAR),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ConvertImageDtype(torch.float32),
                                      transforms.Normalize(*NORM)])
    return train_transform


# SimCLR Transformations
def simclr_transforms(image_size: Union[int, tuple], jitter: tuple = (0.4, 0.4, 0.2, 0.1),
                      p_blur: float = 1.0, p_solarize: float = 0.0,
                      normalize: list = [[0.485],[0.229]], translation=True, scale=(0.4, 0.8), gray_scale=True,
                      sol_threshold=0.42, p_jitter=0.8, vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    """
    Returns a composition of transformations for SimCLR training.

    Args:
        image_size (Union[int, tuple]): The size of the output image. If int, the output image will be square with sides of length `image_size`. If tuple, the output image will have dimensions `image_size[0]` x `image_size[1]`.
        jitter (tuple, optional): Tuple of four floats representing the range of random color jitter. Defaults to (0.4, 0.4, 0.2, 0.1).
        p_blur (float, optional): Probability of applying Gaussian blur. Defaults to 1.0.
        p_solarize (float, optional): Probability of applying random solarization. Defaults to 0.0.
        normalize (list, optional): List of two lists representing the mean and standard deviation for image normalization. Defaults to [[0.485],[0.229]].
        translation (bool, optional): Whether to apply small random translation. Defaults to True.
        scale (tuple, optional): Tuple of two floats representing the range of random scale for random resized crop. Defaults to (0.4, 0.8).
        gray_scale (bool, optional): Whether the input image is in gray scalr or not. Defaults to True.
        sol_threshold (float, optional): Threshold for random solarization. Defaults to 0.42.
        p_jitter (float, optional): Probability of applying color jitter. Defaults to 0.8.
        vertical_flip (bool, optional): Whether to apply random vertical flip. Defaults to False.
        p_horizontal_flip (float, optional): Probability of applying random horizontal flip. Defaults to 0.5.
        rotate (bool, optional): Whether to apply random rotation. Defaults to False.

    Returns:
        torchvision.transforms.Compose: A composition of transformations.
    """
    trans_list = []
    image_size = pair(image_size)

    # Add small translation. This was added after TINC paper
    if translation:
        trans_list.append(transforms.RandomAffine(0,(0.05,0.05), fill=0, interpolation=transforms.InterpolationMode.BILINEAR))
    if rotate:
        trans_list.append(transforms.RandomRotation(rotate, interpolation=transforms.InterpolationMode.BILINEAR))

    trans_list += [transforms.RandomResizedCrop(image_size, scale=scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), #TODO antialias is introduced on 0.15 not stable yet
                  transforms.RandomHorizontalFlip(p=p_horizontal_flip),
                  transforms.ConvertImageDtype(torch.float32)]

    trans_list.append(transforms.RandomApply([transforms.ColorJitter(*jitter)], p=p_jitter))

    if vertical_flip:
        trans_list.append(transforms.RandomVerticalFlip(p=0.5))
    #If image is not grayscale add RandomGrayscale
    if not gray_scale:
        trans_list.append(transforms.RandomGrayscale(p=0.2))
    # Turn off blur for small images
    if image_size[0]<=32:
        p_blur = 0.0
    # Add Gaussian blur
    if p_blur==1.0:
        trans_list.append(transforms.GaussianBlur(image_size[0]//20*2+1))
    elif p_blur>0.0:
        trans_list.append(transforms.RandomApply([transforms.GaussianBlur(image_size[0]//20*2+1)], p=p_blur))
    # Add RandomSolarize
    if p_solarize>0.0:
        trans_list.append(transforms.RandomSolarize(sol_threshold, p=p_solarize))

    if normalize:
        trans_list.extend([transforms.Normalize(*normalize)])
    
    return transforms.Compose(trans_list)


# A wrapper that performs returns two augmented images
class TwoTransform(object):
    """Applies data augmentation two times."""

    def __init__(self, base_transform, sec_transform = None, temporal=False):
        self.base_transform = base_transform
        self.sec_transform = base_transform if sec_transform is None else sec_transform
        self.temporal = temporal

    def __call__(self, x):
        x1 = self.base_transform(x)
        if self.temporal:
            return x1
        x2 = self.sec_transform(x)
        return x1,x2

class VICReg_augmentaions(TwoTransform):
    def __init__(self, image_size, normalize=[[0.485],[0.229]], gray_scale=True, temporal=False, translation=True, scale=(0.4, 0.8)):
        trans1 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale)
        
        trans2 = simclr_transforms(image_size,
                                   p_blur = 0.8,
                                   p_solarize = 0.2,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale)
        
        super().__init__(trans1, trans2, temporal=temporal)

def Barlow_augmentaions(image_size, normalize=[[0.485],[0.229]], gray_scale=True, translation=False, p_blur=0.8, scale=(0.08, 1.0), 
                        temporal=False, sol_threshold=0.42, jitter: tuple = (0.4, 0.4, 0.2, 0.1), p_solarize=0.2, p_jitter=0.8, 
                        vertical_flip=False, p_horizontal_flip=0.5, rotate=False):
    if temporal:        
        trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        return trans1
    
    # This is for vanilla Barlow twins for OCT
    trans1 = simclr_transforms(image_size,
                                   p_blur = p_blur,
                                   p_solarize = 0.0,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter = p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    trans2 = simclr_transforms(image_size,
                                   p_blur = max(0.0, 1.0-p_blur),
                                   p_solarize = p_solarize,
                                   normalize = normalize,
                                   translation = translation,
                                   scale = scale,
                                   gray_scale = gray_scale,
                                   sol_threshold = sol_threshold,
                                   jitter = jitter,
                                   p_jitter=p_jitter,
                                   vertical_flip=vertical_flip,
                                   p_horizontal_flip=p_horizontal_flip,
                                   rotate=rotate)
        
    return [trans1, trans2]

def Monai2DTransforms(transforms, image_keys=["image"],rand_pick=0.0):
    transf = monai.transforms.Compose([shiftBscans2D(keys = ["image"], prob=rand_pick, allow_missing_keys=False),
                                       monai.transforms.RandLambdad(keys = image_keys, func = lambda x: transforms(x),prob=1.0),
                                       monai.transforms.ToTensord(keys = image_keys + ["label"], track_meta=False)])
    return transf

def Monai2DESSLContrastTransforms(transforms, transforms_contr):
    # This is for essl test wth 2d images
    transf = monai.transforms.Compose([monai.transforms.CopyItemsd(keys=["image"], times=1, names=["image_contr1"], allow_missing_keys=False),
                     monai.transforms.CopyItemsd(keys=["image"], times=1, names=["image_contr2"], allow_missing_keys=False),
                     monai.transforms.RandLambdad(keys = ["image"], func = lambda x: transforms(x),prob=1.0),
                     monai.transforms.RandLambdad(keys = ["image_contr1"], func = lambda x: transforms_contr[0](x),prob=1.0),
                     monai.transforms.RandLambdad(keys = ["image_contr2"], func = lambda x: transforms_contr[1](x),prob=1.0),
                     monai.transforms.ToTensord(keys = ["image","image_contr1","image_contr2","label"], track_meta=False)])
    return transf

def Monai2DAUGSELFContrastTransforms(transforms):
    # This is for augself test wth 2d images
    transf = monai.transforms.Compose([monai.transforms.RandLambdad(keys = ["image"], func = lambda x: transforms(x),prob=1.0),
                     monai.transforms.ToTensord(keys = ["image","label"], track_meta=False)])
    return transf

class shiftBscans2D(monai.transforms.MapTransform, monai.transforms.Randomizable):
    # Randomly pick a Bscan
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool, prob: float = 1.0) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob = np.clip(prob, 0.0, 1.0)
    def __call__(self, data):
        if self.R.random() < self.prob:
            for key in self.keys: #self.key_iterator(data):
                item = data[key]
                rand_num = self.R.randint(0,item.shape[-1])
                data[key] = item[..., rand_num]
        return data

def Monai2DContrastTransforms(transforms):
    # This is for standart contrastive learning without temporal inputs
    transf = monai.transforms.Compose([monai.transforms.CopyItemsd(keys=["image"], times=1, names=["image_1"], allow_missing_keys=False),
                     monai.transforms.RandLambdad(keys = ["image"], func = lambda x: transforms[0](x),prob=1.0),
                     monai.transforms.RandLambdad(keys = ["image_1"], func = lambda x: transforms[1](x),prob=1.0),
                     monai.transforms.ToTensord(keys = ["image","image_1","label"], track_meta=False)])

    return transf