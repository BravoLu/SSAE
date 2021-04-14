import torch
import os
import argparse
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from advGAN import AdvGAN_Attack
from models import *
from transforms import get_opts

use_cuda=True
image_nc=3
epochs = 60
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

parser = argparse.ArgumentParser(description="AdvGAN Adversarial Attack")
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--dir', default='', type=str)
parser.add_argument('--gpu', default='0', type=str)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
model_name =  args.model
data_name = args.dataset
data_dir = args.dir

print("model_name: {}".format(model_name))
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

targeted_model = init_model(model_name, '../classification/target/{}_{}.pth'.format(data_name, model_name))
models_path = '../logs/classification/{}_{}_advGAN/'.format(data_name, model_name)
if not os.path.exists(models_path):
    os.makedirs(models_path)
targeted_model = targeted_model.cuda()
targeted_model = nn.DataParallel(targeted_model)
targeted_model.eval()
model_num_labels = 101
cfg = get_opts(data_name)
# MNIST train dataset and dataloader declaration
if data_name == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'data/cifar10', train=True, transform=cfg['transform_train'], download=True))
elif data_name == 'imagenette':
    trainset = ImageFolder(os.path.join(args.dir, 'imagenette2/train'), transform=cfg['transform_train'])
elif data_name == 'caltech101':
    trainset = ImageFolder(os.path.join(args.dir, 'Caltech101/train'), transform=cfg['transform_train'])

train_loader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX, models_path)

advGAN.train(train_loader, epochs)
