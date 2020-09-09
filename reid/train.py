# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2020-06-10 14:36:37
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2020-06-24 10:11:37

import argparse
import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from tqdm import tqdm
import pdb
import sys

from utils import *
from loss import *
from datasets import *
from metrics import *
from models import *
from transforms import *
# from mask import GradCam


lr=1e-4
EPOCHS = 40
delta = 0.1
alpha = 0.0001
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_gallery_feats(model, gallery_loader):
    gallery_features = OrderedDict()
    gallery_labels = OrderedDict()
    for _, inputs in tqdm(enumerate(gallery_loader), desc='Extract gallery features...'):
        imgs, vids, fnames = inputs[0].to(device), inputs[1].to(device), inputs[-1]
        model.eval()
        feats = model(imgs, is_training=False)
        feats = nn.functional.normalize(feats, dim=1, p=2)
        feats = feats.data.cpu()
        for fname, feat, vid in zip(fnames, feats, vids):
            gallery_features[fname] = feat
            gallery_labels[fname] = vid.item()

    return gallery_features, gallery_labels

def get_query_feats(model, query_loader):
    query_features = OrderedDict()
    # query_labels = OrderedDict()
    for _, inputs in tqdm(enumerate(query_loader), desc='Extract query features...'):
        imgs, fnames = inputs[0].to(device), inputs[-1]
        model.eval()
        feats = model(imgs, is_training=False)
        feats = nn.functional.normalize(feats, dim=1, p=2)
        feats = feats.data.cpu()
        for fname, feat in zip(fnames, feats):
            query_features[fname] = feat

    # query_features = torch.stack([query_features[q[-1]] for q in query])
    return query_features

def get_adv_query_feats(model, generator, query_loader, logs, delta):
    mkdir_if_missing(logs)
    query_adv_features = OrderedDict()
    query_labels = OrderedDict()
    raw_imgs, adv_imgs = [], []
    generator.eval()
    for idx, inputs in tqdm(enumerate(query_loader), desc='Extract adv query features...'):
        raw_img, vids, fnames = inputs[0].to(device), inputs[1].to(device), inputs[-1]
        with torch.no_grad():
            perturbations, saliency_map = generator(raw_img)#.data.cpu()
        perturbations = perturbations.detach().cpu()
        saliency_map = saliency_map.detach().cpu()
        adv_img = raw_img.cpu() + batch_clamp(delta, perturbations)

        raw_imgs.append(raw_img.cpu())
        adv_imgs.append(adv_img.cpu())

        raw_feats = model(raw_img.to(device), is_training=False)
        raw_feats = nn.functional.normalize(raw_feats, dim=1, p=2)
        raw_feats = raw_feats.data.cpu()
        adv_feats = model(adv_img.to(device), is_training=False)
        adv_feats = nn.functional.normalize(adv_feats, dim=1, p=2)
        adv_feats = adv_feats.data.cpu()

        for fname, feat, vid in zip(fnames, adv_feats, vids):
            query_adv_features[fname] = feat
            query_labels[fname] = vid.item()

        if idx == 0:
            save_image(saliency_map[0], '%s/mask.jpg'%logs)
            vis_imgs = torch.cat([adv_img], dim=0)
            tensor2img(vis_imgs, mean=mean, std=std)
            save_image(vis_imgs[0], '%s/perturbed.jpg'%logs)
            # save_imgs(vis_imgs[0], '%s/perturbed.jpg'%(logs))

    raw_imgs = torch.cat(raw_imgs, dim=0)
    adv_imgs = torch.cat(adv_imgs, dim=0)

    return query_adv_features, query_labels, raw_imgs, adv_imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adversarial Attack In ReID Task.')
    parser.add_argument('--dataset', type=str, default='Market1501')
    parser.add_argument('--target', type=str, default='pcb')
    parser.add_argument('--ckpt', type=str, default='../targets/reid/pcb_market1501.pth')
    parser.add_argument('--gpu', type=str, default='3,7')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--net', type=str, default='SSAE')
    parser.add_argument('--saliency', action='store_true')
    parser.add_argument('--dir', type=str, default='')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    mode = 'saliency' if args.saliency else 'baseline'
    logs = '../logs/reid/%s_%s_%s'%(args.dataset, args.target, mode)
    mkdir_if_missing(logs)
    logger = Logger(log_dir=logs)
    # dataset = Market1501()
    dataset = globals()[args.dataset](root=args.dir)
    cfg = get_opts(args.target)
    train_loader = DataLoader(
        Preprocessor(dataset.train, training=True, transform=cfg['transform_train']),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    query_loader = DataLoader(
        Preprocessor(dataset.query, training=False, transform=cfg['transform_test']),
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, training=False, transform=cfg['transform_test']),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    features = OrderedDict()
    labels = OrderedDict()
    device = torch.device('cuda')

    # model
    # generator = EncoderDecoder().to(device)
    generator = globals()[args.net]().to(device)
    generator = nn.DataParallel(generator)
    if args.saliency:
        EPOCHS = 150
        try:
            generator.load_state_dict(torch.load('../logs/reid/%s_%s_baseline/Best_G.pth'%(args.dataset, args.target)))
        except:
            print('You must train the symmetric saliency-based auto-encoder without saliency map first')
            sys.exit()
    num_classes = {
        'Market1501': 751,
        'CUHK03': 767,
    }
    # generator.load_state_dict(torch.load('../logs/reid/pcb_baseline_0.05/Best_G.pth'))
    target_model = init_model(name=args.target, pre_dir=args.ckpt, num_classes=num_classes[args.dataset])
    if args.target == 'pcb':
        target_model = nn.DataParallel(PCB_test(target_model)).to(device)
    else:
        target_model = nn.DataParallel(target_model).to(device)

    logger.write('baseline target: %s lr: %s alpha: %f delta %f\n'%(args.target, args.lr, alpha, delta))
    min_mAP = 1.0
    # optimizer
    optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    # caculate gallery features in advance
    gallery_features, gallery_labels = get_gallery_feats(target_model, gallery_loader)
    # loss
    query_features = get_query_feats(target_model, query_loader)
    raw_mAP, raw_rank = cal_mAP_cmc(query_features, gallery_features, dataset.query, dataset.gallery, args.dataset)

    logger.write('raw mAP: {:.2f} raw rank-1: {:.2f}\n'.format(raw_mAP, raw_rank[0]))

    cosine_loss = CosineLoss().to(device)
    mse_loss = nn.MSELoss(reduction='sum').to(device)
    target_model.eval()
    for epoch in range(EPOCHS):
        if args.saliency:
            stats = ['angular_loss', 'norm_loss', 'frobenius_loss', 'loss']
        else:
            stats = ['loss', 'angular_loss']
        meters_trn = {stat: AverageMeter() for stat in stats}
        generator.train()
        for batch_idx, inputs in tqdm(enumerate(train_loader)):
            raw_imgs, fnames = inputs[0].to(device), inputs[-1]

            # perturb images
            perturbations, saliency_map = generator(raw_imgs)
            perturbations = batch_clamp(delta, perturbations)
            if args.saliency:
                adv_imgs = raw_imgs + perturbations * saliency_map
            else:
                adv_imgs =  raw_imgs + perturbations

            # extract features from imgs and adv_imgs
            raw_feats = target_model(raw_imgs, is_training=False)
            raw_norms = torch.norm(raw_feats, dim=1)
            raw_feats = nn.functional.normalize(raw_feats, dim=1, p=2)
            adv_feats = target_model(adv_imgs, is_training=False)
            adv_norms = torch.norm(adv_feats, dim=1)
            adv_feats = nn.functional.normalize(adv_feats, dim=1, p=2)

            angular_loss = cosine_loss(raw_feats, adv_feats)
            loss = angular_loss
            if args.saliency:
                norm_loss = mse_loss(raw_norms, adv_norms)
                frobenius_loss = torch.norm(saliency_map, dim=(1,2), p=2).sum()
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

        query_adv_features, query_labels, raw_imgs, adv_imgs = get_adv_query_feats(target_model, generator, query_loader, logs, delta)
        mAP, rank = cal_mAP_cmc(query_adv_features, gallery_features, dataset.query, dataset.gallery, args.dataset)
        tensor2img(raw_imgs, mean=mean, std=std)
        tensor2img(adv_imgs, mean=mean, std=std)
       # ssim_score = ssim(raw_imgs, adv_imgs, val_range=1.0, size_average=True)
        logger.write('{} | Baseline | Target: {}| Epoch: {} | mAP: {} | rank-1: {} | \n'.format(args.dataset, args.target, epoch, mAP, rank[0]))
        torch.save(generator.state_dict(), '%s/Best_G.pth'%(logs))
            # print('generator checkpoints saved!')

        # print('---- loss information ----')
        #print('cos loss:%f mssim loss:%f adv loss:%f attack loss:%f'%(sum(losses)/len(losses), sum(ml_losses)/len(ml_losses), sum(adv_losses)/len(adv_losses), sum(attack_losses)/len(attack_losses)))
        for s in stats:
            logger.write('%s: %f |'%(s, meters_trn[s].avg))
        logger.write('\n')

