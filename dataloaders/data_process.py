import random

import torch
import numpy as np
from scipy import ndimage
from torchvision.transforms import RandomChoice, RandomEqualize, ColorJitter


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, run_type="train"):
        self.run_type = run_type
        self.trans = RandomChoice([
            RandomEqualize(0.5),
            ColorJitter(brightness=0.5),
            ColorJitter(contrast=0.5),
            ColorJitter(saturation=0.5),
            ColorJitter(hue=0.5)
        ])

    def __call__(self, image, label):
        if self.run_type == "train":
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
            image = torch.from_numpy(image).unsqueeze(0)
            image = self.trans(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)
        image = image / 255
        label = torch.from_numpy(label.astype(np.uint8))
        return image, label
