# %%
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tifffile
import cv2
import torchvision.transforms.functional as TF
import random
from PIL import Image
# %%
class Imagefolder(datasets.ImageFolder):
    def __init__(self, img_dir, size= (100, 100), resize = (480, 384), transform=None, preprocess = None):
        super(Imagefolder, self).__init__(img_dir)
        self.transform = transform
        self.resize = resize
        self.size = size
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample[0]
        lbl = sample[1]
        img = Image.open(path)
        # if img.ndim ==2:
        #     img = img[..., np.newaxis]
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # shape = img.shape
        # if (self.size[0] > shape[0]) | (self.size[1] > shape[1]):
        #     raise ValueError(f'Size={self.size} > {shape[0]},{shape[1]}')
        # img = img.astype('float32')
        data_transforms = transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Resize((self.resize[0],self.resize[1]))
                                              ])
        img = data_transforms(img)
        if self.preprocess is not None:
            img = self.preprocess(img)
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
                                              # transforms.RandomResizedCrop(size=size, scale = (0.08,1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              # RotationTransform(angles=[0, 90, 180, 270])
                                              # transforms.RandomRotation(degrees = (0,180))
                                              ])
        return data_transforms

    def get_dataset(self, resize = (480, 384), transform = True, preprocess = None):
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
