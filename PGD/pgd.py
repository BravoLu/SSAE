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
from tqdm import tqdm

from utils import *
from datasets import *
from models import *
from metrics import *
from transforms import *

eps = 0.1

from advertorch.attacks import L2PGDAttack, FGSM
def PGDAttack(model, test_loader):
    
    # adversary = L2PGDAttack(model, eps=1.0, eps_iter=1*2/40, nb_iter=40,rand_init=False, targeted=False, clip_min=-1000, clip_max=1000)
    #adversary = CarliniWagnerL2Attack(model, num_classes=10)
    adversary = FGSM(model, eps=0.1, clip_min=-1000, clip_max=1000)
    all_raw_imgs, all_adv_imgs = [], []
    model.eval()
    correct, total = 0, 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
        # if batch_idx > 10:
        #     break
        raw_imgs, targets = inputs.cuda(), targets.cuda()
        adv_imgs = adversary.perturb(raw_imgs)

        #tensor2img(adv_imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #save_image(adv_imgs, "tmp.png")
        all_raw_imgs.append(raw_imgs.cpu())
        all_adv_imgs.append(adv_imgs.cpu())

        predicts = model(adv_imgs)
        _, predicts = predicts.max(1)
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()

    all_raw_imgs = torch.cat(all_raw_imgs, dim=0)
    all_adv_imgs = torch.cat(all_adv_imgs, dim=0)
    acc = 100 * correct/total
    return acc, all_raw_imgs, all_adv_imgs

def test(opt):
    cfg = get_opts(opt.dataset)
    print(cfg)
    logs = '../logs/classification/%s_%s_%s'%(opt.dataset, opt.target, 'PGD')
    mkdir_if_missing(logs)
    logger = Logger(log_dir=logs)
    print_information(logger, opt)
    if opt.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root=args.dir, train=False, transform=cfg['transform_test'], download=True)
    elif opt.dataset == 'imagenette':
        testset = ImageFolder(os.path.join(args.dir, 'imagenette2/val'), transform=cfg['transform_test'])
    elif opt.dataset == 'caltech101':
        testset = ImageFolder(os.path.join(args.dir, 'Caltech101/test'), transform=cfg['transform_test'])

    test_loader = DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_classes = {
        'imagenette': 10,
        'cifar10': 10,
        'caltech101': 101,
    }

    target_weights = os.path.join('target/%s_%s.pth'%(opt.dataset, opt.target))
    target_model = init_model(opt.target, target_weights, num_classes=num_classes[opt.dataset])
    target_model = nn.DataParallel(target_model).cuda()

    #raw_acc, _, _ = evaluate(test_loader, target_model, None, logs, eps, mean=cfg['mean'], std=cfg['std'])
    #logger.write('Raw Accuracy: {:2f}\n'.format(raw_acc))
    #adv_acc, raw_imgs, adv_imgs = evaluate(test_loader, target_model, generator, logs, eps, opt.saliency, mean=cfg['mean'], std=cfg['std'])
    print("target model:%s"%opt.target)
    print("dataset:%s"%opt.dataset)
    adv_acc, raw_imgs, adv_imgs = PGDAttack(target_model, test_loader)

    logger.write('Adv Accuray: {:2f}\n'.format(adv_acc))
    tensor2img(raw_imgs, cfg['mean'], cfg['std'])
    tensor2img(adv_imgs, cfg['mean'], cfg['std'])
    print(raw_imgs.size(), adv_imgs.size())
    ssim_score = ssim(raw_imgs, adv_imgs, val_range=1.0)
    psnr_score = psnr(raw_imgs.cpu().numpy(), adv_imgs.cpu().numpy())
    ms_ssim_score = msssim(raw_imgs, adv_imgs, val_range=1.0)
    print('ssim:{:3f} ms-ssim:{:3f} psnr:{:4f}'.format(ssim_score, ms_ssim_score, psnr_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Test phase: Adversarial Attack In Classification Task.')
    parser.add_argument('--dataset', type=str, default='imagenette', choices=('cifar10', 'imagenette', 'caltech101'))
    parser.add_argument('--target', type=str, default='resnet', choices=('efficientnet', 'densenet', 'resnet','googlenet', 'mobilenet'))
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--net', type=str, default='SSAE')
    parser.add_argument('--dir', type=str, default='', help='the dir of dataset')
    parser.add_argument('--saliency', action='store_true', help='if use saliency or not')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test(args)
