import torch
import argparse
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import models
from models import *
from metrics import *
from transforms import get_opts
from tqdm import tqdm
from torchvision.utils import save_image
import os
use_cuda=True
image_nc=3
batch_size = 128
gen_input_nc = image_nc

parser = argparse.ArgumentParser('AdvGAN')
parser.add_argument('--dir', type=str, default='')
args = parser.parse_args()

def tensor2img(tensors, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
    for c in range(3):
        tensors.data[:, c, :, :] = tensors.data[:, c, :, :] * std[c] + mean[c]

# model_name = 'resnet'
# model_name = 'googlenet'
# model_name = 'efficientnet'
# model_name = 'mobilenet'
model_name = 'densenet'
data_name = 'caltech101'

#Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
os.environ["VISIBLE_CUDA_DEVICE"] = "5"
#load the pretrained model
#targeted_model = init_model('resnet', '../targets/classification/imagenette_resnet.pth')
targeted_model = init_model(model_name, '../classification/target/{}_{}.pth'.format(data_name, model_name))
#targeted_model = init_model('resnet', '../targets/classification/cifar10_resnet.pth')
targeted_model = targeted_model.cuda()
targeted_model.eval()
#load the generator of adversarial examples

#pretrained_generator_path = './models/netG_epoch_20.pth'
pretrained_generator_path = '../logs/classification/{}_{}_advGAN/netG_epoch_60.pth'.format(data_name, model_name)
#pretrained_generator_path = '../logs/classification/cifar10_resnet_advGAN/netG_epoch_20.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).cuda()
#pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
#cfg = get_opts('imagenette')
cfg = get_opts(data_name)
if data_name == 'imagenette':
    testset = ImageFolder(os.path.join(args.dir, 'imagenette2/val'), transform=cfg['transform_test'])
elif data_name == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'cifar10'), train=False, transform=cfg['transform_test'], download=True)
elif data_name == 'caltech101':
    testset = ImageFolder(os.path.join(args.dir, 'Caltech101/test'), transform=cfg['transform_test'])

test_dataloader = DataLoader(
        testset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
)
# test adversarial examples in MNIST testing dataset
num_correct = 0
adv_imgs = []
raw_imgs = []
for i, data in tqdm(enumerate(test_dataloader)):
    test_img, test_label = data
    test_img, test_label = test_img.cuda(), test_label.cuda()
    with torch.no_grad():
        perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #perturbation = torch.clamp(perturbation, -0.1, 0.1)
    #print(perturbation)
    #import pdb
    #pdb.set_trace()
    adv_img = perturbation + test_img
    #adv_img = test_img
    #adv_img = torch.clamp(adv_img, 0, 1)

    tensor2img(adv_img, mean=cfg['mean'], std=cfg['std'])
    tensor2img(test_img, mean=cfg['mean'], std=cfg['std'])
    adv_imgs.append(adv_img.cpu())
    raw_imgs.append(test_img.cpu())

    with torch.no_grad():
        pred_lab = torch.argmax(targeted_model(adv_img)[1],1)
    if i == 1:
       save_image(adv_img[:8], 'test.png')
       save_image(test_img[:8], 'ori.png')
    num_correct += torch.sum(pred_lab==test_label,0)

raw_imgs = torch.cat(raw_imgs, dim=0)
adv_imgs = torch.cat(adv_imgs, dim=0)
print(raw_imgs.size(), adv_imgs.size())
ssim_score = ssim(raw_imgs, adv_imgs, val_range=1.0)
ms_ssim_score = msssim(raw_imgs, adv_imgs, val_range=1.0)
psnr_score = psnr(raw_imgs.cpu().numpy(), adv_imgs.cpu().numpy())
print('ms_ssim_score:{}'.format(ms_ssim_score))
print('psnr_score:{}'.format(psnr_score))
print('ssim_score:{}'.format(ssim_score))
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(testset)))

