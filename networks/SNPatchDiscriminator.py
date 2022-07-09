import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from .blocks import Conv2dBlock, UpConv2dBlock

class SNPatchDiscriminator(nn.Module):
    def __init__(self, cfg, num_layer):
        super(SNPatchDiscriminator, self).__init__()

        input_nc = cfg['input_nc'] + cfg['lab_dim']
        ndf = cfg['ndf']
        self.num_layer = num_layer

        # self.dis1 = Conv2dBlock(input_nc, ndf, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        # self.dis2 = Conv2dBlock(ndf, ndf * 2, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        # self.dis3 = Conv2dBlock(ndf * 2, ndf * 4, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        # self.dis4 = Conv2dBlock(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        # self.dis5 = Conv2dBlock(ndf * 4, ndf * 4, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')
        # self.dis6 = Conv2dBlock(ndf * 4, 3, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')

        dis_layers = [Conv2dBlock(input_nc, ndf, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
        for i in range(self.num_layer-2):
            mult = 2 ** i
            if mult < 4:
                dis_layers += [Conv2dBlock(ndf * mult, ndf * mult * 2, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
            else:
                mult = 4
                dis_layers += [Conv2dBlock(ndf * mult, ndf * mult, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]
        dis_layers += [Conv2dBlock(ndf * 4, 3, kernel_size=5, stride=2, padding=2, norm='sn', activation='lrelu')]

        self.dis_model = nn.Sequential(*dis_layers)

    def forward(self, input):
        # d1 = self.dis1(input)
        # d2 = self.dis2(d1)
        # d3 = self.dis3(d2)
        # d4 = self.dis4(d3)
        # d5 = self.dis5(d4)
        # d6 = self.dis6(d5)

        d = self.dis_model(input)

        return d