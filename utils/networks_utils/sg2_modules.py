import math
import random

import numpy as np
import torch
from utils.networks_utils.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from torch import nn, autograd
from torch.nn import functional as F


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grads_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = [x.pow(2).view(x.shape[0], -1).sum(1).mean() for x in grads_real]
    grad_penalty = sum(grad_penalty) / len(grad_penalty)

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    if len(grad.shape) == 2:
        grad = grad.unsqueeze(1)

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def g_path_regularize_spade(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grads = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    B, S, _, _ = grads[0].shape
    grads = [x.view(B, S, -1) for x in grads]
    grads = torch.cat(grads, -1)
    # print('grads', grads.shape)

    path_lengths = torch.sqrt(grads.pow(2).sum(1).mean(1))
    # print('path_lengths', path_lengths.shape)

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    # print('path_mean', path_mean.shape)

    path_penalty = (path_lengths - path_mean).pow(2).mean()
    # print('path_penalty', path_penalty.shape)

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def make_noise_uniform(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.rand(batch, latent_dim, device=device) * 2 - 1

    noises = torch.rand(n_noise, batch, latent_dim, device=device).unbind(0) * 2 - 1

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, replicate_pad=False):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.replicate_pad = replicate_pad

        self.pad = pad

    def forward(self, input):
        if self.replicate_pad:
            if type(self.pad) == tuple and len(self.pad) == 2:
                pad_tuple = (self.pad[0], self.pad[1], self.pad[0], self.pad[1])
            else:
                pad_tuple = (self.pad, self.pad, self.pad, self.pad)
            input = F.pad(input, pad_tuple, mode='replicate')
            out = upfirdn2d(input, self.kernel, pad=(0, 0))
        else:
            out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class SirenConv1x1(nn.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.out_f = out_f
        self.b = 1 / in_f if self.is_first else np.sqrt(6 / in_f) / w0

        self.linear = torch.nn.Linear(in_f, out_f)
        torch.nn.init.uniform_(self.linear.weight, -self.b, self.b)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, C)
        out = self.linear(x)
        out = out.reshape(B, H, W, self.out_f)
        out = out.permute(0, 3, 1, 2)
        return out + .5 if self.is_last else self.w0 * out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, replicate_pad=False,
            bias_init=0, lr_mul=1., activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size).div_(lr_mul)
        )
        self.scale = (1 / math.sqrt(in_channel * kernel_size ** 2)) * lr_mul
        self.lr_mul = lr_mul

        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.replicate_pad = replicate_pad
        self.activation = activation

        # self.i=0

    def forward(self, input, dilation=1):

        pad = self.padding * dilation
        if self.replicate_pad:
            input = F.pad(input, (pad, pad, pad, pad), mode='replicate')
            pad = 0

        bias = self.bias * self.lr_mul if self.bias is not None else None
        if self.activation:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                stride=self.stride,
                padding=pad,
                dilation=dilation
            )
            out = fused_leaky_relu(out, bias)
        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=bias,
                stride=self.stride,
                padding=pad,
                dilation=dilation
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConvTranspose2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, bias=True,
            replicate_pad=False, bias_init=0
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # self.padval = padval
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.replicate_pad = replicate_pad

        # self.i=0

    def forward(self, input):
        opad = self.output_padding
        pad = self.padding

        if self.replicate_pad:
            input = F.pad(input, [opad, opad, opad, opad], mode='replicate')
            input = F.pad(input, [pad, pad, pad, pad], mode='replicate')

            opad = 0
            pad = 0

        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=pad,
            output_padding=opad
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):

        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ConstantInputRect(nn.Module):
    def __init__(self, channel, h, w):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, h, w))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


# ============ Normalization

class AdaTexSpade(nn.Module):
    def __init__(self, num_features, segm_tensor, style_dim, kernel_size=1, eps=1e-4):
        super().__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)

        self.segm_tensor = nn.Parameter(segm_tensor, requires_grad=False)
        n_segmchannels = self.segm_tensor.shape[1]
        in_channel = style_dim + n_segmchannels

        self.style_conv = EqualConv2d(
            in_channel,
            num_features,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, input, style):
        B, C, H, W = input.shape
        out = self.norm_layer(input)

        sB, sC = style.shape
        style = style[..., None, None]
        style = style.expand(sB, sC, H, W)
        segm_tensor = self.segm_tensor.expand(sB, *self.segm_tensor.shape[1:])
        style = torch.cat([style, segm_tensor], dim=1)
        gammas = self.style_conv(style)

        out = out * gammas
        return out


class AdaIn(nn.Module):
    def __init__(self, num_features, style_dim, eps=1e-4):
        super().__init__()
        self.num_features = num_features
        self.weight = self.bias = None
        self.norm_layer = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        self.modulation = EqualLinear(style_dim, num_features, bias_init=1)

    def forward(self, input, style):
        B, C, H, W = input.shape
        out = self.norm_layer(input)
        gammas = self.modulation(style)
        gammas = gammas[..., None, None]
        out = out * gammas
        return out


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style, return_s=False, input_is_in_s_space=False, weights_delta=None):
        batch, in_channel, height, width = input.shape

        if not input_is_in_s_space:
            style = self.modulation(style)

        style_to_return = style
        style = style.view(batch, 1, in_channel, 1, 1)
        if weights_delta is None:
            weight = self.scale * self.weight * style
        else:
            weight = self.weight * (1 + weights_delta)
            weight = self.scale * (weight * style)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )

            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        if return_s:
            return out, style_to_return
        else:
            return out


class SimplifyedModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.weight
        style = self.modulation(style)
        input = input * style[..., None, None] * self.scale

        if self.upsample:
            out = F.conv_transpose2d(input, weight.transpose(0, 1), padding=0, stride=2)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            out = F.conv2d(input, weight, padding=0, stride=2)
        else:
            out = F.conv2d(input, weight, padding=self.padding)

        if self.demodulate:
            weight_scale = weight * self.scale
            demod = torch.rsqrt(
                (weight_scale.unsqueeze(0) * style[..., None, None].unsqueeze(1)).pow(2).sum([2, 3, 4]) + 1e-8)

            out = out * demod[..., None, None]

        return out


class SpadeModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualConv2d(style_dim, in_channel, 1, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.weight
        style = self.modulation(style)
        input = input * style * self.scale

        if self.upsample:
            out = F.conv_transpose2d(input, weight.transpose(0, 1), padding=0, stride=2)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            out = F.conv2d(input, weight, padding=0, stride=2)

        else:
            out = F.conv2d(input, weight, padding=self.padding)

        if self.demodulate:
            weight_scale = weight * self.scale

            if self.downsample:
                style_pad = F.pad(style, (0, self.padding, 0, self.padding), 'replicate')
                demod_map = torch.rsqrt(
                    F.conv2d(style_pad.pow(2), weight_scale.pow(2), padding=0, stride=2) + 1e-8)  # * self.scale
            elif self.upsample:
                denom_map = torch.ones_like(style[:, :1])
                denom_weight = torch.ones_like(weight_scale[:1, :1])
                denom_vals = F.conv_transpose2d(denom_map, denom_weight.transpose(0, 1), padding=0, stride=2)

                demod_map = F.conv_transpose2d(style.pow(2), weight_scale.transpose(0, 1).pow(2), padding=0,
                                               stride=2)
                demod_map = demod_map * (self.kernel_size ** 2 / denom_vals)
                demod_map = torch.rsqrt(demod_map + 1e-8)
                demod_map = demod_map[:, :, :-1, :-1]
            else:
                style_pad = F.pad(style, (self.padding, self.padding, self.padding, self.padding), 'replicate')
                demod_map = torch.rsqrt(F.conv2d(style_pad.pow(2), weight_scale.pow(2)) + 1e-8)  # * self.scale

            out = out * demod_map

        return out


# ============ StyledConv

class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            simple_modulation=False
    ):
        super().__init__()

        mconv = SimplifyedModulatedConv2d if simple_modulation else ModulatedConv2d

        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, return_s=False, input_is_in_s_space=False, weights_delta=None):

        if return_s:
            out, s_vec = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                                   weights_delta=weights_delta)
        else:
            out = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                            weights_delta=weights_delta)

        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        if return_s:
            return out, s_vec
        else:
            return out


class StyledConvAInp(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            add_input=None,
            ainp_trainable=False,
            simple_modulation=False
    ):
        super().__init__()

        self.add_input = add_input
        if add_input is not None:
            in_channel = in_channel + add_input.shape[1]
            self.add_input = nn.Parameter(add_input, requires_grad=ainp_trainable)

        mconv = SimplifyedModulatedConv2d if simple_modulation else ModulatedConv2d
        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, return_s=False, input_is_in_s_space=False, weights_delta=None):
        if self.add_input is not None:
            B = input.shape[0]
            ainp = self.add_input.repeat(B, 1, 1, 1)
            input = torch.cat([input, ainp], dim=1)

        if return_s:
            out, s_vec = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                                   weights_delta=weights_delta)
        else:
            out = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                            weights_delta=weights_delta)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        if return_s:
            return out, s_vec
        else:
            return out


class SpadeTestStyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        mconv = SpadeModulatedConv2d

        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        _, _, H, W = input.shape
        style = style[..., None, None].repeat(1, 1, H, W)

        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class SpadeStyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        mconv = SpadeModulatedConv2d

        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class SpadeTestStyledConvAInp(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            add_input=None,
            ainp_trainable=False
    ):
        super().__init__()

        self.add_input = add_input
        if add_input is not None:
            in_channel = in_channel + add_input.shape[1]
            self.add_input = nn.Parameter(add_input, requires_grad=ainp_trainable)

        mconv = SpadeModulatedConv2d
        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        _, _, H, W = input.shape
        style = style[..., None, None].repeat(1, 1, H, W)

        if self.add_input is not None:
            B = input.shape[0]
            ainp = self.add_input.repeat(B, 1, 1, 1)
            input = torch.cat([input, ainp], dim=1)

        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class StyledConv1x1ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim,
                 demodulate=True):
        super().__init__()

        self.conv1 = StyledConv(in_channel, out_channel, 1, style_dim, demodulate=demodulate)
        self.conv2 = StyledConv(out_channel, out_channel, 1, style_dim, demodulate=demodulate)

        self.skip = ConvLayer(in_channel, out_channel, 1, bias=False)

    def forward(self, input, latent, noise=None):
        if type(noise) == list and len(noise) == 2:
            noise1 = noise[0]
            noise2 = noise[1]
        else:
            noise1 = noise
            noise2 = noise

        if latent.ndim == 3:
            latent1 = latent[:, 0]
            latent2 = latent[:, 1]
        else:
            latent1 = latent
            latent2 = latent

        out = self.conv1(input, latent1, noise=noise1)
        out = self.conv2(out, latent2, noise=noise2)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# ============ ToRGB


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1],
                 simple_modulation=False):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        mconv = SimplifyedModulatedConv2d if simple_modulation else ModulatedConv2d
        self.conv = mconv(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False, alfa=1., return_s=False, input_is_in_s_space=False,
                weights_delta=None):

        if return_s:
            out, s_vec = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                                   weights_delta=weights_delta)
        else:
            out = self.conv(input, style, return_s=return_s, input_is_in_s_space=input_is_in_s_space,
                            weights_delta=weights_delta)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = alfa * out + skip

        out_list = [out]
        if return_delta:
            out_list.append(delta)
        if return_s:
            out_list.append(s_vec)

        if len(out_list) == 1:
            return out_list[0]
        else:
            return out_list


class SpadeTestToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        mconv = SpadeModulatedConv2d
        self.conv = mconv(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False, alfa=1.):
        _, _, H, W = input.shape
        style = style[..., None, None].repeat(1, 1, H, W)

        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = alfa * out + skip

        if return_delta:
            return out, delta
        else:
            return out


class SpadeToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        mconv = SpadeModulatedConv2d
        self.conv = mconv(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False, alfa=1.):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = alfa * out + skip

        if return_delta:
            return out, delta
        else:
            return out


class SpadeToRGBFPB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        mconv = SpadeModulatedConv2d
        self.conv = mconv(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def _split(self, features, gap_size=1, only_collapse=False):
        """
        Add a zero filled gap to a featuremap

        For example 4x(4+2) featuremap becomes 4x(4+1+2) with a gap in-between

        :param features: features of style tensor
        :param gap_size: num of rows in the gap
        :param only_collapse: if True, omits the gap and only collapses previously added gap
        :return: feature tensor with a gap added
        """
        if features is None:
            return features

        _, _, H, W = features.shape

        body_bottom_ind = W
        face_top_ind = H - W // 2

        body_features = features[:, :, :body_bottom_ind]
        face_features = features[:, :, face_top_ind:]

        gap = torch.zeros_like(body_features[:, :, :gap_size])

        if only_collapse:
            with_gap = torch.cat([body_features, face_features], dim=2)
        else:
            with_gap = torch.cat([body_features, gap, face_features], dim=2)

        return with_gap

    def forward(self, input, style, skip=None, return_delta=False, alfa=1., onlyface=False):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:

            if not onlyface:
                skip = self._split(skip)
            skip = self.upsample(skip)

            if not onlyface:
                skip = self._split(skip, only_collapse=True)
            delta = out
            out = alfa * out + skip

        if return_delta:
            return out, delta
        else:
            return out


# ============ Other


class SineConv1x1(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            is_first=False,
            omega_0=30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_channel = in_channel
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, 1, 1)
        )

        if is_first:
            a = -1 / self.in_channel
        else:
            a = -np.sqrt(6 / self.in_channel) / self.omega_0
        b = -a

        self.scale = np.abs(b - a) * math.sqrt(1 / 12)

        spectral_tex = torch.load('/Vol1/dbstore/datasets/a.grigorev/gent/smplx_spectral_texture_norm.pth').cuda()
        spectral_tex_inp = spectral_tex[:, 2:100]
        self.spectral_tex_mask = ((spectral_tex ** 2).sum(dim=1) > 0)[0]
        self.sin_input = None

    def forward(self, x):
        weight = self.weight * self.scale
        out = F.conv2d(x, weight)
        mask = self.spectral_tex_mask
        out = torch.sin(self.omega_0 * out)
        return out


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            replicate_pad=False,
            lr_mul=1,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), replicate_pad=replicate_pad))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
                replicate_pad=replicate_pad,
                lr_mul=lr_mul,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=None, replicate_pad=False, lr_mul=1):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3,
                               replicate_pad=replicate_pad, lr_mul=lr_mul, blur_kernel=blur_kernel)
        self.conv2 = ConvLayer(in_channel, out_channel, 3,
                               downsample=True, replicate_pad=replicate_pad, lr_mul=lr_mul, blur_kernel=blur_kernel)

        self.skip = ConvLayer(
            in_channel, out_channel, 1,
            downsample=True, activate=False, bias=False,
            replicate_pad=replicate_pad, lr_mul=lr_mul, blur_kernel=blur_kernel
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


# ==================== Old

class SimplifyedModulatedConv2d_BUG(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        weight = self.weight
        style = self.modulation(style)
        input = input * style[..., None, None]  # * self.scale

        if self.upsample:
            out = F.conv_transpose2d(input, weight.transpose(0, 1), padding=0, stride=2)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            out = F.conv2d(input, weight, padding=0, stride=2)
        else:
            out = F.conv2d(input, weight, padding=self.padding)

        if self.demodulate:
            weight_scale = weight  # * self.scale
            demod = torch.rsqrt(
                (weight_scale.unsqueeze(0) * style[..., None, None].unsqueeze(1)).pow(2).sum([2, 3, 4]) + 1e-8)

            out = out * demod[..., None, None]

        return out


class StyledConv_BUG(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            simple_modulation=False
    ):
        super().__init__()

        mconv = SimplifyedModulatedConv2d_BUG if simple_modulation else ModulatedConv2d

        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledConvAInp_BUG(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
            add_input=None,
            ainp_trainable=False,
            simple_modulation=False
    ):
        super().__init__()

        self.add_input = add_input
        if add_input is not None:
            in_channel = in_channel + add_input.shape[1]
            self.add_input = nn.Parameter(add_input, requires_grad=ainp_trainable)

        mconv = SimplifyedModulatedConv2d_BUG if simple_modulation else ModulatedConv2d
        self.conv = mconv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        if self.add_input is not None:
            B = input.shape[0]
            ainp = self.add_input.repeat(B, 1, 1, 1)
            input = torch.cat([input, ainp], dim=1)

        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB_BUG(nn.Module):
    def __init__(self, in_channel, style_dim, out_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1],
                 simple_modulation=False):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        mconv = SimplifyedModulatedConv2d_BUG if simple_modulation else ModulatedConv2d
        self.conv = mconv(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None, return_delta=False, alfa=1.):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            delta = out
            out = alfa * out + skip

        if return_delta:
            return out, delta
        else:
            return out
