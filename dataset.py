import os
from typing import Tuple
import numpy as np
import torch

import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode as IMode

from imgproc import center_crop
from imgproc import image2tensor
from imgproc import random_crop
from imgproc import random_horizontally_flip
from imgproc import random_rotate

__all__ = ["BaseDataset", "CustomDataset", "Avg_Crop_Dataset", "Normalize", "Denormalize"]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1)

def gen_denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
  
    denormalized_tensors = tensors.clone()
    
    for c in range(3):
        denormalized_tensors[:, c].mul_(std[c]).add_(mean[c])
    
    return torch.clamp(denormalized_tensors, 0, 1)

class Crop_Image(Dataset):
    def __init__(self, hr, transform, num_crops=20):
        super(Crop_Image, self).__init__()
        
        self.transform = transform
        if self.transform:
            self.hr_img = self.transform(hr)

        c, h, w = self.hr_img.shape
        new_h = 224
        new_w = 224
    
        self.lr_img_patches = []
        self.hr_img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            hr_patch = self.hr_img[ :, top: top + new_h, left: left + new_w]
            self.hr_img_patches.append(hr_patch)

    def get_patch(self, idx):
        
        hr_patch = self.hr_img_patches[idx]
        
        return hr_patch
    

class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, img):
        # r_img: C x H x W (numpy)

        d_img = (img - self.mean) / self.var

        return d_img
    
class Denormalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, img):
        # Perform denormalization on the normalized image
        # Assuming img is a numpy array of the form C x H x W
        d_img = (img * self.var) + self.mean
        return d_img

class Avg_Crop_Dataset(Dataset):
    """The basic data set loading function only needs to prepare high-resolution image data.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, num_crops: int) -> None:
        super(Avg_Crop_Dataset, self).__init__()

        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        self.num_crops = num_crops        

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((56,56), interpolation=IMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])
        
        Img = Crop_Image(hr, 
            transform=transforms.Compose([transforms.ToTensor()]), #정규화 넣어줄지말지 결정
            num_crops=self.num_crops)
        
        hr_list = []
        lr_list = []
        for i in range(self.num_crops):
            
            hr_patch = Img.get_patch(i)
            lr_patch = self.lr_transforms(hr_patch)
            hr_list.append(hr_patch)
            lr_list.append(lr_patch)

        return lr_list, hr_list

    def __len__(self) -> int:
        return len(self.filenames)
        
class BaseDataset(Dataset):
    """The basic data set loading function only needs to prepare high-resolution image data.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(BaseDataset, self).__init__()
        lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
        hr_image_size = (image_size, image_size)
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        # Low-resolution images and high-resolution images have different processing methods.
        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(hr_image_size),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ])
            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(lr_image_size, interpolation=IMode.BICUBIC),
                transforms.ToTensor()
            ])

        elif mode == "valid":
            self.hr_transforms = transforms.Compose([
                transforms.CenterCrop(hr_image_size),
                transforms.ToTensor()
            ])
            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(lr_image_size, interpolation=IMode.BICUBIC),
                transforms.ToTensor()
            ])
        elif mode == "original_size_valid":
          
            self.hr_transforms = transforms.Compose([
                transforms.Lambda(lambda img: img.resize(((img.size[0] // 4)*4, (img.size[1] // 4)*4), Image.BICUBIC)),
                transforms.ToTensor()  
            ])
            self.lr_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(lambda img: img.resize((img.size[0] // 4, img.size[1] // 4), Image.BICUBIC)),
                transforms.ToTensor()  
            ])

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        hr = Image.open(self.filenames[index])

        hr = self.hr_transforms(hr)
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)


class CustomDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(CustomDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and low-resolution folder
        # under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_dir_path = os.path.join(dataroot, f"LRunknownx{upscale_factor}")
        hr_dir_path = os.path.join(dataroot, f"HR")
        self.filenames = os.listdir(lr_dir_path)
        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in self.filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in self.filenames]

        self.image_size = image_size  # HR image size.
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = Image.open(self.lr_filenames[index])
        hr = Image.open(self.hr_filenames[index])

        # Data enhancement methods.
        if self.mode == "train":
            lr, hr = random_crop(lr, hr, self.image_size, self.upscale_factor)
            lr, hr = random_rotate(lr, hr, 90)
            lr, hr = random_horizontally_flip(lr, hr, 0.5)
        else:
            lr, hr = center_crop(lr, hr, self.image_size, self.upscale_factor)

        # `PIL.Image` image data is converted to `Tensor` format data.
        lr = image2tensor(lr)
        hr = image2tensor(hr)

        return lr, hr

    def __len__(self) -> int:
        return len(self.filenames)
