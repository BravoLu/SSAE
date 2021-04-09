import argparse
import torch
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pdb
import numpy as np
from torchvision.utils import save_image

from utils import *
from datasets import *
from models import *
from metrics import *
from transforms import *
from train import get_gallery_feats, get_adv_query_feats

Imagenet_stddev = [0.229, 0.224, 0.225]
Imagenet_mean = [0.485, 0.456, 0.406]
eps = 0.1


def test(opt):
    dataset = globals()[opt.dataset](opt.dir)
    # dataset = VeRi776(root='/home/share/zhihui/VeRi')
    cfg = get_opts(opt.target)
    # print(cfg)
    query_loader = DataLoader(
        Preprocessor(dataset.query, training=False, transform=cfg['transform_test']),
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, training=False, transform=cfg['transform_test']),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    generator = globals()[opt.net]().cuda()
    generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(torch.load(opt.g))
    num_classes = {
        'Market1501': 751,
        'CUHK03': 767,
    }
    target_model = init_model(name=opt.target, pre_dir=opt.ckpt, num_classes=num_classes[opt.dataset])
    if opt.target == 'pcb':
        target_model = nn.DataParallel(PCB_test(target_model)).cuda()
    else:
        target_model = nn.DataParallel(target_model).cuda()

    target_model.eval()

    gallery_features, _ = get_gallery_feats(target_model, gallery_loader)
    query_adv_features, _, raw_imgs, adv_imgs  = get_adv_query_feats(target_model, generator, query_loader, opt.log, eps)

    tensor2img(adv_imgs, mean=Imagenet_mean, std=Imagenet_stddev)
    tensor2img(raw_imgs, mean=Imagenet_mean, std=Imagenet_stddev)
    ssim_score = ssim(raw_imgs, adv_imgs, val_range=1.0)
    mAP, rank = cal_mAP_cmc(query_adv_features, gallery_features, dataset.query, dataset.gallery, opt.dataset)
    psnr_score = psnr(raw_imgs.cpu().numpy(), adv_imgs.cpu().numpy())
    ms_ssim_score = msssim(raw_imgs, adv_imgs, val_range=1.0)
    # ssim_score, psnr_score, ms_ssim_score = 0, 0, 0
    print('perturbed mAP:{} rank-1:{} ssim:{} ms-ssim:{} psnr:{}'.format(mAP, rank[0], ssim_score, ms_ssim_score, psnr_score))
    save_image(adv_imgs[0], '%s/mAP_%.2f_ssim_%.2f_psnr_%.2f_ms-ssim_%.2f.jpg'%(opt.log, mAP, ssim_score, psnr_score, ms_ssim_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Test Mode')
    parser.add_argument('--target', type=str, default='pcb')
    parser.add_argument('--ckpt', type=str, default='../targets/reid/pcb_market1501.pth')
    parser.add_argument('--dataset', type=str, default='Market1501')
    parser.add_argument('-g', type=str, default='../logs/reid/pcb_baseline_0.05/Best_G.pth')
    parser.add_argument('--net', type=str, default='SSAE')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--log', type=str, default='')
    parser.add_argument('--saliency', action='store_true')
    parser.add_argument('--dir', type=str, default='')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    test(opt)
