import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .G1 import G1
from .G2 import G2
from .G3 import G3

##########################################
class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.G1 = G1(cfg)
        self.G2 = G2(cfg)
        self.G3 = G3(cfg)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # 'nearest', 'bilinear'

    def forward(self, gt, input, segmap, mask):
        gt_G1 = F.interpolate(gt, size=(64, 64), mode='bilinear')
        gt_G2 = F.interpolate(gt, size=(128, 128), mode='bilinear')
        gt_G3 = F.interpolate(gt, size=(256, 256), mode='bilinear')

        input_G1 =  F.interpolate(input, size=(64, 64), mode='bilinear')
        fake_G1 = self.G1(input_G1, segmap, mask)
        mask_fake_G1 = self.masked_fake(gt_G1, fake_G1, mask)
        input_G2 = self.next_img(gt_G2, fake_G1, mask)

        fake_G2 = self.G2(input_G2, segmap, mask)
        mask_fake_G2 = self.masked_fake(gt_G2, fake_G2, mask)
        input_G3 = self.next_img(gt_G3, fake_G2, mask)

        fake_G3 = self.G3(input_G3, segmap, mask)
        mask_fake_G3 = self.masked_fake(gt_G3, fake_G3, mask)

        return [gt_G1, gt_G2, gt_G3], [input_G1, input_G2, input_G3], [mask_fake_G1, mask_fake_G2, mask_fake_G3], [fake_G1, fake_G2, fake_G3]

    def masked_fake(self, img, fake, mask):
        mask = F.interpolate(mask, size=fake.size()[2:], mode='nearest')
        combined = mask * fake + (1. - mask) * img
        return combined

    def next_img(self, img, prev_fake, mask):
        fake = self.up(prev_fake)
        mask = F.interpolate(mask, size=fake.size()[2:], mode='nearest')
        combined = mask * fake + (1. - mask) * img

        return combined