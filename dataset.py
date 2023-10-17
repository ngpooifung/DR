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
class Imagefolder(datasets.ImageFolder):
    def __init__(self, img_dir, resize = 336, transform=None, preprocess = True, clip_csv = None):
        super(Imagefolder, self).__init__(img_dir)
        self.transform = transform
        self.resize = resize
        self.preprocess = preprocess
        self.clip_csv = clip_csv
        if clip_csv is not None:
            self.csv = pd.read_csv(clip_csv)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample[0]
        lbl = sample[1]
        img = Image.open(path)
        if self.clip_csv is not None:
            text = self.csv['text'][self.csv['label'] == lbl].item()
            name = os.path.split(path)[1]
            subject,date,eye,_ = name.split('-')
            subject = subject[4:]
            date = date.split('@')[0]
            if eye[0] == 'L':
                eye = 'left eye'
            else:
                eye = 'right eye'
            text = f"Subject {subject}'s {eye} is {text}"
        # if img.ndim ==2:
        #     img = img[..., np.newaxis]
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # shape = img.shape
        # if (self.size[0] > shape[0]) | (self.size[1] > shape[1]):
        #     raise ValueError(f'Size={self.size} > {shape[0]},{shape[1]}')
        # img = img.astype('float32')
        data_transforms = transforms.Compose([Resize((self.resize, int(self.resize*1.25)),interpolation=BICUBIC),
                                              # CenterCrop(self.resize),
                                              _convert_image_to_rgb,
                                              ToTensor(),
                                              Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                              ])
        if self.preprocess:
            img = data_transforms(img)
        if self.transform is not None:
            img = self.transform(img)

        return (img, text) if self.clip_csv is not None else (img, sample)

# %%
class Modeldataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform(size):
        data_transforms = transforms.Compose([
                                              # transforms.RandomResizedCrop(size=size, scale = (0.6,1.0)),
                                              transforms.RandomHorizontalFlip(p = 0.5),
                                              transforms.RandomRotation(degrees = 5),
                                              # RandomAffine(degrees = 5),
                                              # ColorJitter(0.01, 0.01),
                                              GaussianBlur((7,9), sigma = (0.1, 2.0)),
                                              ])
        return data_transforms

    def get_dataset(self, resize = 336, transform = True, preprocess = True, clip_csv = None):
        if transform:
            dataset = Imagefolder(img_dir = self.root_folder, resize = resize, transform = self.get_transform(resize), preprocess = preprocess, clip_csv = clip_csv)
        else:
            dataset = Imagefolder(img_dir = self.root_folder, resize = resize, preprocess = preprocess, clip_csv = clip_csv)

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
