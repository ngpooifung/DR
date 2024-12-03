# %%
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tifffile
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, GaussianBlur, ColorJitter, RandomAffine
import random
from PIL import Image
import os
import pandas as pd
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
# %%
def claheimg(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_img)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab_img = cv2.merge(lab_planes)
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

class Imagefolder(datasets.ImageFolder):
    def __init__(self, img_dir, resize = 336, transform=None, preprocess = True):
        super(Imagefolder, self).__init__(img_dir)
        self.transform = transform
        self.resize = resize
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample[0]
        lbl = sample[1]
        # img = Image.open(path)
        img = cv2.imread(path)
        img = Image.fromarray(claheimg(img))
        data_transforms = transforms.Compose([
                                              Resize(self.resize, interpolation=BICUBIC),
                                              CenterCrop((self.resize, int(self.resize*1.25))),
                                              _convert_image_to_rgb,
                                              ToTensor(),
                                              Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                              ])
        if self.preprocess:
            img = data_transforms(img)
        if self.transform is not None:
            img = self.transform(img)

        return (img, sample)

# %%
class Modeldataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(size):
        data_transforms = transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.RandomRotation(degrees = 5),
                                              # RandomAffine(degrees = 30, translate = (0.25, 0.25)),
                                              # ColorJitter(0.01, 0.01),
                                              GaussianBlur((7,9), sigma = (0.1, 2)),
                                              # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                              ])
        return data_transforms

    def get_dataset(self, resize = 336, transform = True, preprocess = True):
        if transform:
            dataset = Imagefolder(img_dir = self.root_folder, resize = resize, transform = self.get_transform(resize), preprocess = preprocess)
        else:
            dataset = Imagefolder(img_dir = self.root_folder, resize = resize, preprocess = preprocess)

        return dataset


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def _convert_image_to_rgb(image):
    return image.convert("RGB")
