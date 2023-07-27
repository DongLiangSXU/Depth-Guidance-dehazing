import torch
from PIL import Image
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision.utils as vutils

from depth_fuse import Tfuse_Net
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from functools import partial
from base_networks import *
from depth_finetune import NormDepth
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)

        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out
def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block_ffa(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_ffa, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
        # self.ealayer=External_attention(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        # res=self.ealayer(res)
        res += x
        return res
class multi_Net(nn.Module):

    def __init__(self, **kwargs):
        super(multi_Net, self).__init__()

        self.depth_finetune=NormDepth(iskt=False)
        self.mix1 = Mix(m=-0.6)
        self.Tfuse_second = Tfuse_Net()



    def forward(self, h):

        depth,depth_=self.depth_finetune(h)
        depthmap=torch.cat((depth_,depth_,depth_),dim=1)
        y=self.mix1(depthmap,h)
        y1=self.Tfuse_second(y,h)

        dehazemap=y1


        return dehazemap,depth_
