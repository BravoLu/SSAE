# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import torch.optim as optim
from copy import deepcopy

from utils import batch_clamp
from utils import rand_init_delta

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, norm='bn', n_blocks=6):
        super(Generator, self).__init__()

        n_downsampling = n_upsampling = 2
        use_bias = norm == 'in'
        norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
        begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
        for i in range(n_upsampling):
            up_layers.append([])
        # ngf
        begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
        # 2ngf, 4ngf
        for i in range(n_downsampling):
            mult = 2**i
            down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
        # 4ngf
        mult = 2**n_downsampling
        for i in range(n_blocks):
            res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
        # 2ngf, ngf
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            up_layers[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

        end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.l1 = nn.Sequential(*begin_layers)
        self.l2 = nn.Sequential(*down_layers)
        self.l3 = nn.Sequential(*res_layers)
        self.l4_1 = nn.Sequential(*up_layers[0])
        self.l4_2 = nn.Sequential(*up_layers[1])
        self.l5 = nn.Sequential(*end_layers)

        # self.set_requires_grad(self)  # this line cause bug


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, inputs):
        out = self.l1(inputs)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4_1(out)
        out = self.l4_2(out)
        out = self.l5(out)
        return out

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        for i in range(2):
            conv_block += [nn.ReflectionPad2d(1)]
            conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias)), norm_layer(dim)]
            if i < 1:
                conv_block += [nn.ReLU(True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class SSAE(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, norm='bn', n_blocks=6):
        super(SSAE, self).__init__()

        n_downsampling = n_upsampling = 2
        use_bias = norm == 'in'
        norm_layer = nn.BatchNorm2d if norm == 'bn' else nn.InstanceNorm2d
        begin_layers, down_layers, res_layers, up_layers, end_layers = [], [], [], [], []
        for i in range(n_upsampling):
            up_layers.append([])
        # ngf
        begin_layers = [nn.ReflectionPad2d(3), SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)), norm_layer(ngf), nn.ReLU(True)]
        # 2ngf, 4ngf
        for i in range(n_downsampling):
            mult = 2**i
            down_layers += [SpectralNorm(nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias)), norm_layer(ngf*mult*2), nn.ReLU(True)]
        # 4ngf
        mult = 2**n_downsampling
        for i in range(n_blocks):
            res_layers += [ResnetBlock(ngf*mult, norm_layer, use_bias)]
        # 2ngf, ngf
        for i in range(n_upsampling):
            mult = 2**(n_upsampling - i)
            up_layers[i] += [SpectralNorm(nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]

        end_layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        end_layers2 = [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, kernel_size=7, padding=0), nn.Tanh()]

        self.encoder_1 = nn.Sequential(*begin_layers)
        self.encoder_2 = nn.Sequential(*down_layers)
        self.encoder_3 = nn.Sequential(*res_layers)

        self.perturb_decoder_1 = nn.Sequential(*up_layers[0])
        self.perturb_decoder_2 = nn.Sequential(*up_layers[1])
        self.perturb_decoder_3 = nn.Sequential(*end_layers)

        self.mask_decoder_1 = deepcopy(self.perturb_decoder_1)
        self.mask_decoder_2 = deepcopy(self.perturb_decoder_2)
        self.mask_decoder_3 = nn.Sequential(*end_layers2)
        # self.set_requires_grad(self)  # this line cause bug


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x):
        # encoding
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        latency = self.encoder_3(x)

        # decoding
        delta = self.perturb_decoder_1(latency)
        delta = self.perturb_decoder_2(delta)
        delta = self.perturb_decoder_3(delta)

        mask = self.mask_decoder_1(latency)
        mask = self.mask_decoder_2(mask)
        mask = self.mask_decoder_3(mask)
        mask = mask - torch.min(torch.min(mask, dim=2)[0], dim=2)[0].unsqueeze(2).unsqueeze(2)
        mask = mask / torch.max(torch.max(mask, dim=2)[0], dim=2)[0].unsqueeze(2).unsqueeze(2)

        return delta, mask


