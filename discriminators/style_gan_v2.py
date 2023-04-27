import math

import torch
from torch import nn

from utils.networks_utils.sg2_modules import EqualLinear, ConvLayer, ResBlock


class Discriminator(nn.Module):
    def __init__(
            self,
            image_size,
            input_channels=3,
            channel_multiplier=4,
            activation_layer=2,
            late_input_channels=0,
            blur_kernel=[1, 3, 3, 1],
            k_first=1,
            replicate_pad=False,
            std_features=False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.image_size = image_size
        self.channel_multiplier = channel_multiplier
        self.activation_layer = activation_layer
        self.blur_kernel = blur_kernel

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

        convs = [ConvLayer(input_channels, channels[image_size], k_first, replicate_pad=replicate_pad)]

        log_size = int(math.log(image_size, 2))

        in_channel = channels[image_size] + late_input_channels

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, replicate_pad=replicate_pad))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        if std_features:
            in_channel += 1
        self.final_conv = ConvLayer(in_channel, channels[4], 3, replicate_pad=replicate_pad)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        self.std_features = std_features

    def forward(self, data_dict):
        input = data_dict['disc_input']
        activations = []

        out = input
        for conv in self.convs:
            out = conv(out)
            activations.append(out)
        batch, channel, height, width = out.shape

        if self.std_features:

            group = min(batch, self.stddev_group)
            stddev = out.view(
                group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
            )
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        activations.append(out)
        out = out.view(batch, -1)

        out = self.final_linear(out)

        # return results
        out_dict = {
            'score': out,
            'activations': activations[self.activation_layer],
        }
        return out_dict
