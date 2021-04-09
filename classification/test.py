import argparse
import torch
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pdb
import numpy as np
from torchvision.utils import save_image
import torchvision
from torchvision.datasets import ImageFolder

from utils import *
from datasets import *
from models import *
from metrics import *
from transforms import *

eps = 0.1

def test(opt):
    cfg = get_opts(opt.dataset)
    print(cfg)
    mode = 'saliency' if args.saliency else 'baseline'
    logs = '../logs/classification/%s_%s_%s_v2'%(opt.dataset, opt.target, mode)
    mkdir_if_missing(logs)
    logger = Logger(log_dir=logs)
    print_information(logger, opt)
    if opt.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'data/cifar10'), train=False, transform=cfg['transform_test'], download=True)
    elif opt.dataset == 'imagenette':
        testset = ImageFolder(os.path.join(args.dir, 'imagenette2/val'), transform=cfg['transform_test'])

    test_loader = DataLoader(
        testset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    generator_weights = os.path.join(logs, 'Best_G.pth')
    generator = globals()[opt.net]().cuda()
    generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(torch.load(generator_weights))
    # generator.load_state_dict(torch.load(generator_weights))
    # generator = torch.nn.DataParallel(generator)
    # torch.save(generator.module.state_dict(), 'attacker.pth')
    print("save successfully!")
    num_classes = {
        'imagenette': 10,
        'cifar10': 10,
    }

    target_weights = os.path.join('target/%s_%s.pth'%(opt.dataset, opt.target))
    target_model = init_model(opt.target, target_weights, num_classes=num_classes[args.dataset])
    target_model = nn.DataParallel(target_model).cuda()

    #raw_acc, _, _ = evaluate(test_loader, target_model, None, logs, eps, mean=cfg['mean'], std=cfg['std'])
    #logger.write('Raw Accuracy: {:2f}\n'.format(raw_acc))
    adv_acc, raw_imgs, adv_imgs = evaluate(test_loader, target_model, generator, logs, eps, opt.saliency, mean=cfg['mean'], std=cfg['std'])
    logger.write('Adv Accuray: {:2f}\n'.format(adv_acc))
    tensor2img(raw_imgs, cfg['mean'], cfg['std'])
    tensor2img(adv_imgs, cfg['mean'], cfg['std'])
    ssim_score = ssim(raw_imgs, adv_imgs, val_range=1.0)
    psnr_score = psnr(raw_imgs.cpu().numpy(), adv_imgs.cpu().numpy())
    ms_ssim_score = msssim(raw_imgs, adv_imgs, val_range=1.0)
    print('ssim:{:3f} ms-ssim:{:3f} psnr:{:4f}'.format(ssim_score, ms_ssim_score, psnr_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Test phase: Adversarial Attack In Classification Task.')
    parser.add_argument('--dataset', type=str, default='imagenette', choices=('cifar10', 'imagenette'))
    parser.add_argument('--target', type=str, default='resnet', choices=('efficientnet', 'densenet', 'resnet','googlenet', 'mobilenet'))
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--net', type=str, default='SSAE')
    parser.add_argument('--dir', type=str, default='', help='the dir of dataset')
    parser.add_argument('--saliency', action='store_true', help='if use saliency or not')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test(args)
