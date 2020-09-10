# -*- coding: utf-8 -*-

from __future__ import absolute_import

from torchvision.transforms import *
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
import math

Imagenet_mean = [0.485, 0.456, 0.406]
Imagenet_stddev = [0.229, 0.224, 0.225]
class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, mean=[0.485, 0.456, 0.406]):
        self.EPSILON = EPSILON
        self.mean = mean

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(0.02, 0.2) * area
            aspect_ratio = random.uniform(0.3, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]

                return img

        return img

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

def get_opts(name):
  # 1.
  base_opt = {}
  if name == 'ide':
    base_opt['transform_train'] = T.Compose([RandomSizedRectCrop(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev), RandomErasing(EPSILON=0)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'densenet121':
    base_opt['transform_train'] = T.Compose([Random2DTranslation(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'mudeep':
    base_opt['transform_train'] = T.Compose([Random2DTranslation(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  # 2.
  elif name == 'aligned':
    base_opt['transform_train'] = T.Compose([Random2DTranslation(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128)), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'pcb':
    base_opt['transform_train'] = T.Compose([T.Resize((384,192), interpolation=3), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(Imagenet_mean, Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((384,192), interpolation=3), T.ToTensor(), T.Normalize(Imagenet_mean, Imagenet_stddev)])
    base_opt['ReID_factor'] = 2
    base_opt['workers'] = 16

  elif name == 'hacnn':
    base_opt['transform_train'] = T.Compose([Random2DTranslation(160, 64), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((160, 64)), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  # 3.
  elif name == 'cam':
    base_opt['transform_train'] = T.Compose([RandomSizedRectCrop(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev), RandomErasing(EPSILON=0.5)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'lsro':
    base_opt['transform_train'] = T.Compose([T.Resize(144, interpolation=3), T.RandomCrop((256,128)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((288,144), interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'hhl':
    base_opt['transform_train'] = T.Compose([RandomSizedRectCrop(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev), RandomErasing(EPSILON=0)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'spgan':
    base_opt['transform_train'] = T.Compose([RandomSizedRectCrop(256, 128), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev), RandomErasing(EPSILON=0)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128), interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])

  elif name == 'sbl':
    base_opt['transform_train'] = T.Compose([T.Resize((256, 128),interpolation=3), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((256, 128),interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
  elif name == 'mgn':
    base_opt['transform_train'] = T.Compose([T.Resize((384, 128),interpolation=3), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
    base_opt['transform_test'] = T.Compose([T.Resize((384, 128),interpolation=3), T.ToTensor(), T.Normalize(mean=Imagenet_mean, std=Imagenet_stddev)])
  return base_opt
