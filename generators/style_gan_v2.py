import math

import torch
from torch import nn

from utils.general_utils import posenc
from utils.networks_utils.sg2_modules import ConstantInput, StyledConv, ToRGB, EqualLinear, ConvLayer, ResBlock, \
    PixelNorm, StyledConvAInp


class RandomEncoder(nn.Module):
    def __init__(self, n_mlp=8, style_dim=512, lr_mlp=0.01):
        super().__init__()
        self.style_dim = style_dim

        self.pixnorm = PixelNorm()

        self.linears = nn.ModuleList()
        self.linears.append(
            EqualLinear(
                style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
            ))

        for i in range(n_mlp):
            self.linears.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

    def forward(self, data_dict, mix_prob=0.9):
        input = data_dict['input_rgb']
        input = torch.randn(input.shape[0], self.style_dim, device=input.device)
        style = self.pixnorm(input)

        for linear in self.linears:
            style = linear(style)
        out_dict = {'style': [style]}
        return out_dict


class Encoder(nn.Module):
    def __init__(
            self,
            image_size,
            input_channels=3,
            style_dim=512,
            channel_multiplier=4,
            late_input_channels=0,
            blur_kernel=[1, 3, 3, 1],
            k_first=1,
            lr_mul=0.01,
            replicate_pad=False,
    ):
        super().__init__()

        self.input_channels = input_channels + 1 + 1
        self.image_size = image_size
        self.channel_multiplier = channel_multiplier
        self.blur_kernel = blur_kernel
        self.style_dim = style_dim

        channels = {
            4: min(128 * channel_multiplier, 512),
            8: min(128 * channel_multiplier, 512),
            16: min(128 * channel_multiplier, 512),
            32: min(128 * channel_multiplier, 512),
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 8 * channel_multiplier,
        }
        self.size2channel = channels
        convs = [ConvLayer(self.input_channels, channels[image_size], k_first,
                           replicate_pad=replicate_pad, lr_mul=lr_mul, blur_kernel=blur_kernel)]

        log_size = int(math.log(image_size, 2))

        in_channel = channels[image_size] + late_input_channels

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, replicate_pad=replicate_pad, lr_mul=lr_mul))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3,
                                    replicate_pad=replicate_pad, lr_mul=lr_mul, blur_kernel=blur_kernel)

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', lr_mul=lr_mul),
            EqualLinear(channels[4], style_dim, lr_mul=lr_mul),
        )

    def forward(self, data_dict):
        input = data_dict['input_rgb']
        segm = data_dict['real_segm']
        input_noise = torch.randn(
            input.shape[0], 1, input.shape[2], input.shape[3], device=input.device
        )
        input = torch.cat([input, segm, input_noise], dim=1)
        activations = []

        out = input
        for conv in self.convs:
            out = conv(out)
            activations.append(out)
        batch, channel, height, width = out.shape

        out = self.final_conv(out)
        activations.append(out)
        out = out.view(batch, -1)

        out = self.final_linear(out)

        # return results
        out_dict = {'style': [out]}
        return out_dict


class Generator(nn.Module):
    def __init__(
            self,
            texture_size,
            style_dim=512,
            channel_multiplier=4,
            output_channels=3,
            blur_kernel=[1, 3, 3, 1],
            ainp_path='xyz_512.pth',
            ainp_scales=[8, 16, 32, 64, 128, 256, 512],
            ainp_posenc=False,
            ainp_train=False,
    ):
        super().__init__()

        self.ainp_train = ainp_train
        if ainp_path is None or ainp_scales is None:
            add_input = None
            ainp_scales = []
        else:
            add_input = torch.load(ainp_path)
            if ainp_posenc:
                add_input = posenc(add_input, 6, 2)

        self.size = texture_size
        self.style_dim = style_dim
        self.output_channels = output_channels

        self.channels = {
            4: min(128 * channel_multiplier, 512),
            8: min(128 * channel_multiplier, 512),
            16: min(128 * channel_multiplier, 512),
            32: min(128 * channel_multiplier, 512),
            64: 128 * channel_multiplier,
            128: 64 * channel_multiplier,
            256: 32 * channel_multiplier,
            512: 16 * channel_multiplier,
            1024: 8 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, out_channel=output_channels, upsample=False)

        self.log_size = int(math.log(texture_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            scale = 2 ** i
            out_channel = self.channels[scale]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            if scale in ainp_scales:
                ainp = torch.nn.functional.interpolate(add_input, size=(scale, scale), mode='bilinear')
                self.convs.append(
                    StyledConvAInp(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, add_input=ainp,
                        ainp_trainable=self.ainp_train
                    )
                )
            else:
                self.convs.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, out_channel=output_channels))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def reset_reqgrad(self):
        if not self.ainp_train:
            for conv in self.convs:
                if hasattr(conv, 'add_input') and conv.add_input is not None:
                    conv.add_input.requires_grad = False

    def forward(
            self,
            data_dict,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            noise=None,
            randomize_noise=True,
            return_mipmaps=False,
            return_noise=False
    ):
        total_convs = len(self.convs) + len(self.to_rgbs) + 2  # +2 for first conv and toRGB

        if 'weights_deltas' not in data_dict:
            weights_deltas = [None] * total_convs
        else:
            weights_deltas = data_dict['weights_deltas']

        styles = data_dict['style']

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation != 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        latent = styles[0].unsqueeze(1).repeat(1, self.n_latent, 1)

        if return_mipmaps:
            mipmaps = []
        else:
            mipmaps = None

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0], weights_delta=weights_deltas[0])
        skip = self.to_rgb1(out, latent[:, 1], weights_delta=weights_deltas[1])

        if mipmaps is not None:
            mipmaps.append(skip)

        i = 1
        weight_idx = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1, weights_delta=weights_deltas[weight_idx])
            out = conv2(out, latent[:, i + 1], noise=noise2, weights_delta=weights_deltas[weight_idx + 1])
            if mipmaps is not None:
                skip, mipmap = to_rgb(out, latent[:, i + 2], skip, return_delta=True,
                                      weights_delta=weights_deltas[weight_idx + 2])
                mipmaps.append(mipmap)
            else:
                skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        # return results
        out_dict = dict(ntexture=image)
        if return_latents:
            out_dict['w_real'] = latent

        if return_mipmaps:
            out_dict['mipmaps'] = mipmaps

        if return_noise:
            out_dict['noise'] = noise

        return out_dict
