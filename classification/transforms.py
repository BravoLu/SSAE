# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-29 16:53:29
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-29 17:01:14
from torchvision.transforms import *
import torchvision.transforms as T 

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_stddev = (0.2023, 0.1994, 0.2010)

def get_opts(name):
    base_opt = {}
    if name == 'cifar10':
        # if use augumentation ?
        base_opt['mean'] = cifar_mean
        base_opt['std'] = cifar_stddev 
        base_opt['transform_train'] = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean=base_opt['mean'], std=base_opt['std'])]) 
        base_opt['transform_test'] = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean=base_opt['mean'], std=base_opt['std'])]) 
    elif name == 'imagenette':
        base_opt['mean'] = cifar_mean
        base_opt['std'] = cifar_stddev 
        base_opt['transform_train'] = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=base_opt['mean'], std=base_opt['std'])]) 
        base_opt['transform_test'] = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=base_opt['mean'], std=base_opt['std'])]) 


    return base_opt 
