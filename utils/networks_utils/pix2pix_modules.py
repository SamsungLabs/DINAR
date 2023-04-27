import functools

import torch
import torch.nn as nn
from utils.networks_utils.op import FusedLeakyReLU

from utils.networks_utils.sg2_modules import EqualConv2d


###############################################################################
# Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class GlobalGeneratorSkip(nn.Module):
    def __init__(
            self,
            input_nc,
            output_nc,
            ngf=64,
            n_downsampling=3,
            n_blocks=9,
            norm_layer=nn.BatchNorm2d,
            final_k=7,
            replicate_pad=False,
            final_act=None,
    ):
        assert (n_blocks >= 0)
        super().__init__()
        activation = FusedLeakyReLU
        padding_type = 'reflect' if not replicate_pad else 'replicate'

        start = [nn.ReflectionPad2d(3), EqualConv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf),
                 activation(ngf)]
        self.start = nn.Sequential(*start)
        self.n_downsampling = n_downsampling

        ### downsample
        downsamples = []
        for i in range(n_downsampling):
            mult = 2 ** i
            downsamples.append(nn.Sequential(EqualConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1,
                                                         replicate_pad=replicate_pad),
                                             norm_layer(ngf * mult * 2), activation(ngf * mult * 2)))
        self.downsamples = nn.ModuleList(downsamples)

        ### resnet blocks
        resblocks = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            resblocks += [
                ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.resblocks = nn.Sequential(*resblocks)

        ### upsample
        upsamples = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsamples.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                           EqualConv2d(2 * ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1,
                                                       padding=1, replicate_pad=replicate_pad),
                                           norm_layer(int(ngf * mult / 2)), activation(int(ngf * mult / 2))))
        self.upsamples = nn.ModuleList(upsamples)

        final = [nn.ReflectionPad2d(final_k // 2), EqualConv2d(ngf, output_nc, kernel_size=final_k, padding=0)]
        if final_act == 'tanh':
            final.append(nn.Tanh())
        elif final_act == 'sigm':
            final.append(nn.Sigmoid())

        self.final = nn.Sequential(*final)

    def forward(self, input):
        x = self.start(input)

        skips = []
        for i in range(self.n_downsampling):
            x = self.downsamples[i](x)
            skips.append(x)

        x = self.resblocks(x)

        for i in range(self.n_downsampling):
            inp = torch.cat([x, skips[-i - 1]], dim=1)
            x = self.upsamples[i](inp)

        x = self.final(x)

        return x


class Start(nn.Module):
    def __init__(self, input_nc, out_nc, norm_layer, activation, pad=True):
        super().__init__()
        self.pad = 3 if pad else 0
        self.conv = EqualConv2d(input_nc, out_nc, kernel_size=7, padding=0)
        self.norm = norm_layer(out_nc)
        self.activation = activation(out_nc)

    def forward(self, x, dilation=1):
        pad = self.pad * dilation
        pad_seq = (pad, pad, pad, pad)
        x = torch.nn.functional.pad(x, pad_seq, mode='reflect')
        x = self.conv(x, dilation=dilation)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, input_nc, out_nc, norm_layer, activation, pad=True):
        super().__init__()
        self.pad = 1 if pad else 0
        self.conv = EqualConv2d(input_nc, out_nc, kernel_size=3, stride=2)
        self.norm = norm_layer(out_nc)
        self.activation = activation(out_nc)

    def forward(self, x, dilation=1):
        pad = self.pad * dilation
        pad_seq = (pad, pad, pad, pad)
        x = torch.nn.functional.pad(x, pad_seq, mode='reflect')
        x = self.conv(x, dilation=dilation)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_nc, out_nc, norm_layer, activation, pad=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pad = 1 if pad else 0
        self.conv = EqualConv2d(input_nc, out_nc, kernel_size=3, stride=1)
        self.norm = norm_layer(out_nc)
        self.activation = activation(out_nc)

    def forward(self, x, dilation=1):
        x = self.upsample(x)
        pad = self.pad * dilation
        pad_seq = (pad, pad, pad, pad)
        x = torch.nn.functional.pad(x, pad_seq, mode='reflect')
        x = self.conv(x, dilation=dilation)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Final(nn.Module):
    def __init__(self, input_nc, out_nc, final_k, pad=True):
        super().__init__()
        self.pad = final_k // 2 if pad else 0
        self.conv = EqualConv2d(input_nc, out_nc, kernel_size=final_k, padding=0)
        self.act = nn.Tanh()

    def forward(self, x, dilation=1):
        pad = self.pad * dilation
        pad_seq = (pad, pad, pad, pad)
        x = torch.nn.functional.pad(x, pad_seq, mode='reflect')

        x = self.conv(x, dilation=dilation)
        x = self.act(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [EqualConv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation(dim)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [EqualConv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
