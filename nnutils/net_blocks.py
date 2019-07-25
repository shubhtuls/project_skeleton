'''
CNN building blocks.
'''

from __future__ import division
from __future__ import print_function
import math

import torch
import torch.nn as nn
import torchvision
import functools


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


## fc layers
def fc(nc_inp, nc_out):
    return nn.Sequential(
        nn.Linear(nc_inp, nc_out),
        torch.nn.modules.normalization.LayerNorm(nc_out, elementwise_affine=False),
        nn.LeakyReLU(0.01,inplace=False)
    )


def fc_stack(nc_inp, nc_out, nlayers):
    modules = []
    for l in range(nlayers):
        modules.append(fc(nc_inp, nc_out))
        nc_inp = nc_out
    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder


## 2D convolution layers
class ScalarMultiply(nn.Module):
    def __init__(self, scale_factor):
        super(ScalarMultiply, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x*self.scale_factor


def conv2d(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        # torch.nn.modules.normalization.GroupNorm(out_planes, out_planes, affine=False),
        # ScalarMultiply(1/out_planes),
        nn.LeakyReLU(0.01,inplace=False)
    )


def deconv2d(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.01,inplace=False)
    )


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='bilinear'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


def upconv2d(in_planes, out_planes, mode='bilinear'):
    upconv = nn.Sequential(
        Upsample(scale_factor=2, mode=mode),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0),
        #torch.nn.modules.normalization.GroupNorm(out_planes, out_planes, affine=False),
        #ScalarMultiply(1/out_planes),
        nn.LeakyReLU(0.01,inplace=False)
    )
    return upconv


def conv2d_stack(nlayers, nc_input=1, nc_max=128, nc_l1=8, nc_step=1):
    ''' Simple 3D encoder with nlayers.
    
    Args:
        nlayers: number of encoder layers
        nc_input: number of input channels
        nc_max: number of max channels
        nc_l1: number of channels in layer 1
        nc_step: double number of channels every nc_step layers
    '''
    modules = []
    nc_output = nc_l1
    for nl in range(nlayers):
        if (nl>=1) and (nl%nc_step==0) and (2*nc_output <= nc_max):
            nc_output *= 2

        modules.append(conv2d(nc_input, nc_output, stride=1))
        nc_input = nc_output
        modules.append(conv2d(nc_input, nc_output, stride=1))
        modules.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

    encoder = nn.Sequential(*modules)
    net_init(encoder)
    return encoder, nc_output


def upconv2d_stack(nlayers, nc_input=128, nc_out=3, nc_min=8, nc_step=1):
    ''' Simple 2D decoder with nlayers.
    
    Args:
        nlayers: number of upconv layers
        nc_input: number of input channels
        nc_out: number of output channels
        nc_min: number of min channels
        nc_step: half the number of channels every nc_step layers
    '''
    modules = []
    nc_output = nc_input//2

    for nl in range(nlayers):
        if (nl>=1) and (nl%nc_step==0) and (nc_output <= 2*nc_min):
            nc_output = nc_output//2

        modules.append(upconv2d(nc_input, nc_output))
        nc_input = nc_output
        modules.append(conv2d(nc_input, nc_output, stride=1))

    modules.append(nn.Conv2d(nc_output, nc_out, kernel_size=3, stride=1, padding=1, bias=True))

    decoder = nn.Sequential(*modules)
    net_init(decoder)
    return decoder


class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, pretrained=True):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            #n = m.out_features
            #m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
            #n = m.in_features
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #xavier
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.ConvTranspose2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #this modified initialization seems to work better, but it's very hacky
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            # Initialize Deconv with bilinear weights.
            base_weights = bilinear_init(m.weight.data.size(-1))
            base_weights = base_weights.unsqueeze(0).unsqueeze(0)
            m.weight.data = base_weights.repeat(m.weight.data.size(0), m.weight.data.size(1), 1, 1)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def bilinear_init(kernel_size=4):
    # Following Caffe's BilinearUpsamplingFiller
    # https://github.com/BVLC/caffe/pull/2213/files
    import numpy as np
    width = kernel_size
    height = kernel_size
    f = int(np.ceil(width / 2.))
    cc = (2 * f - 1 - f % 2) / (2.*f)
    weights = torch.zeros((height, width))
    for y in range(height):
        for x in range(width):
            weights[y, x] = (1 - np.abs(x / f - cc)) * (1 - np.abs(y / f - cc))

    return weights


## Taken from pix2pix code
## https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L532
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


if __name__ == '__main__':
    decoder2d(5, None, 256, use_deconv=True, init_fc=False)
    bilinear_init()
