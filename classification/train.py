# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-10 14:36:37
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-29 17:14:02

import argparse
import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from tqdm import tqdm
import pdb
import torchvision

from utils import *
from loss import *
from datasets import *
from metrics import *
from models import *
from transforms import *
# from mask import GradCam

EPOCHS = 40
delta = 0.1
alpha = 0.0001

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adversarial Attack In Classification Task.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'imagenette'))
    parser.add_argument('--target', type=str, default='resnet', choices=('efficientnet', 'densenet', 'resnet', 'vgg', 'googlenet', 'mobilenet'))
    parser.add_argument('--ckpt', type=str, default='../targets/classification/Baseline_cifar10.pth')
    parser.add_argument('--gpu', type=str, default='3,7')
    parser.add_argument('--dir', type=str, default='', help='the dir of dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--net', type=str, default='SSAE')
    parser.add_argument('--saliency', action='store_true', help='if use saliency map or not')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    mode = 'saliency' if args.saliency else 'baseline'
    logs = '../logs/classification/%s_%s_%s'%(args.dataset, args.target, mode)
    mkdir_if_missing(logs)
    logger = Logger(log_dir=logs)
    # dataset = Market1501()
    cfg = get_opts(args.dataset)

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir,'data/cifar10'), train=True, transform=cfg['transform_train'], download=True)
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'data/cifar10'), train=False, transform=cfg['transform_test'], download=True)
    elif args.dataset == 'imagenette':
        trainset = ImageFolder(os.path.join(args.dir, 'imagenette2/train'), transform=cfg['transform_train'])
        testset = ImageFolder(os.path.join(args.dir, 'imagenette2/val'), transform=cfg['transform_test'])

    train_loader = DataLoader(
        trainset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device('cuda')

    generator = globals()[args.net]().to(device)
    generator = nn.DataParallel(generator)
    if args.saliency:
        EPOCHS = 150
        try:
            generator.load_state_dict(torch.load('../logs/classification/%s_%s_Baseline/Best_G.pth'%(args.dataset, args.target)))
        except:
            print('You must train the symmetric saliency-based auto-encoder without saliency first')
    num_classes = {
        'imagenette': 10,
        'cifar10': 10
    }
    target_model = init_model(args.target, args.ckpt)
    target_model = nn.DataParallel(target_model).to(device)

    print_information(logger, args)
    #logger.write('%s | target: %s | lr: %.5f | alpha: %.5f | delta %.2f\n'%("Saliency Map" if args.saliency else "Baseline", args.target, args.lr, alpha, delta))

    optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    raw_acc, _, _ = evaluate(test_loader, target_model, None, logs, delta, mean=cfg['mean'], std=cfg['std'])
    logger.write('The accuracy before attack: {:2f}\n'.format(raw_acc))

    cosine_loss = CosineLoss().to(device)
    mse_loss = nn.MSELoss(reduction='sum')

    for epoch in range(EPOCHS):
        if args.saliency:
            stats = ['angular_loss', 'norm_loss', 'frobenius_loss', 'loss']
        else:
            stats = ['angular_loss', 'loss']
        meters_trn = {stat: AverageMeter() for stat in stats}
        generator.train()
        for batch_idx, inputs in tqdm(enumerate(train_loader)):
            raw_imgs, targets = inputs[0].to(device), inputs[1].to(device)
            # perturb images
            perturbations, saliency_map = generator(raw_imgs)
            if args.saliency:
                perturbations = batch_clamp(delta, perturbations) * saliency_map
            else:
                perturbations = batch_clamp(delta, perturbations)

            adv_imgs = raw_imgs + perturbations
            # extract features from imgs and adv_imgs
            raw_feats, _ = target_model(raw_imgs)
            raw_norms = torch.norm(raw_feats, dim=1)
            raw_feats = nn.functional.normalize(raw_feats, dim=1, p=2)
            adv_feats, _ = target_model(adv_imgs)
            adv_norms = torch.norm(adv_feats, dim=1)
            adv_feats = nn.functional.normalize(adv_feats, dim=1, p=2)

            angular_loss = cosine_loss(raw_feats, adv_feats)
            loss = angular_loss

            if args.saliency:
                norm_loss = mse_loss(raw_norms, adv_norms)
                frobenius_loss = torch.norm(saliency_map, dim=(1,2)).sum()
                loss += alpha * (norm_loss + frobenius_loss)
                if torch.isnan(frobenius_loss):
                    print("there are nans in frobenius loss")
                    break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in stats:
                v = locals()[k]
                meters_trn[k].update(v.item(), 32)

        # test
        acc, all_raw_imgs, all_adv_imgs = evaluate(test_loader, target_model, generator, logs, delta, use_saliency_map=args.saliency, mean=cfg['mean'], std=cfg['std'])
        logger.write(' Mode: {} | Target dataset: {} | Target model: {}| Epoch: {} | Accuracy: {:2f} |\n'.format(mode, args.dataset, args.target, epoch, acc))
        torch.save(generator.state_dict(), '%s/Best_G.pth'%(logs))

        for s in stats:
            logger.write('%s: %.3f |'%(s, meters_trn[s].avg))
        logger.write('\n')
