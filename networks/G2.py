import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .blocks import CondGatedConv2d, CondTransposeGatedConv2d, Conv2dBlock

##########################################
class G2(nn.Module):
    def __init__(self, cfg):
        super(G2, self).__init__()

        input_nc = cfg['input_nc']
        ngf = cfg['ngf']
        output_nc = cfg['output_nc']
        lab_nc = cfg['lab_dim'] + 1
        g_norm = cfg['G_norm_type']

        # Encoder layers
        self.enc1 = CondGatedConv2d(input_nc, ngf, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm, activation='lrelu')
        self.enc2 = CondGatedConv2d(ngf, ngf * 2, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc3 = CondGatedConv2d(ngf * 2, ngf * 4, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc4 = CondGatedConv2d(ngf * 4, ngf * 4, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc5 = CondGatedConv2d(ngf * 4, ngf * 8, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc6 = CondGatedConv2d(ngf * 8, ngf * 8, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc7 = CondGatedConv2d(ngf * 8, ngf * 16, lab_nc, kernel_size=3, stride=2, padding=1, norm=g_norm, activation='lrelu')
        self.enc8 = CondGatedConv2d(ngf * 16, ngf * 16, lab_nc, kernel_size=3, stride=1, padding=1, dilation=1,
                                    norm=g_norm, activation='lrelu')

        # Decoder layers
        self.dec7 = CondTransposeGatedConv2d(ngf * 16 + ngf * 8, ngf * 16, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec6 = CondTransposeGatedConv2d(ngf * 16 + ngf * 8, ngf * 8, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec5 = CondTransposeGatedConv2d(ngf * 8 + ngf * 4, ngf * 4, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec4 = CondTransposeGatedConv2d(ngf * 4 + ngf * 4, ngf * 2, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec3 = CondTransposeGatedConv2d(ngf * 2 + ngf * 2, ngf, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec2 = CondTransposeGatedConv2d(ngf, ngf, lab_nc, kernel_size=3, stride=1, padding=1, norm=g_norm,
                                   activation='lrelu')
        self.dec1 = Conv2dBlock(ngf, output_nc, kernel_size=3, stride=1, padding=1, norm='none', activation='tanh')

    # In this case, we have very flexible unet construction mode.
    def forward(self, input, segmap, mask):
        # Encoder
        e1 = self.enc1(input, segmap, mask)
        e2 = self.enc2(e1, segmap, mask)
        e3 = self.enc3(e2, segmap, mask)
        e4 = self.enc4(e3, segmap, mask)
        e5 = self.enc5(e4, segmap, mask)
        e6 = self.enc6(e5, segmap, mask)
        e7 = self.enc7(e6, segmap, mask)
        e8 = self.enc8(e7, segmap, mask)

        d7 = self.dec7(e8, segmap, mask, skip=e6)
        d6 = self.dec6(d7, segmap, mask, skip=e5)
        d5 = self.dec5(d6, segmap, mask, skip=e4)
        d4 = self.dec4(d5, segmap, mask, skip=e3)
        d3 = self.dec3(d4, segmap, mask, skip=e2)
        d2 = self.dec2(d3, segmap, mask)
        d1 = self.dec1(d2)

        return d1