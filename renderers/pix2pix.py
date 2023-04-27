"""
Pix2Pix-based renderer implementation.
"""
import torch
from utils.networks_utils.op import FusedLeakyReLU
from torch import nn

from utils.general_utils import to_sigm, to_tanh
from utils.networks_utils.pix2pix_modules import GlobalGeneratorSkip, get_norm_layer
from utils.networks_utils.sg2_modules import EqualConv2d


class Renderer(nn.Module):
    """
    Transform rasterized SMPL-X mesh and neural texture into RGB image.
    """
    def __init__(
            self,
            in_channels_ntex,
            segm_channels,
            ngf=64,
            n_downsampling=3,
            n_blocks=3,
            normalization='batch',
            hardsegm=True,
            replicate_pad=False,
            segm_threshold=0.,
            exponential_transparency=True,
    ):
        """
        Create U-Net like architecture for neural rendering.

        :param in_channels_ntex: Number of channels in the neural texture
        :param segm_channels: Number of segmentation channels (single class / multi class)
        :param ngf: Number of filters in convolution layers
        :param n_downsampling: Number of downsamling blocks
        :param n_blocks: Number of blocks in the middle of U-Net
        :param normalization: Type of normalization layers ["batch", "instance"]
        :param hardsegm: Flag to concat segmentation output with SMPL-X projection
        :param replicate_pad: Flag to use replicated paddings
        :param segm_threshold: Set all values of output segmentation lower than that to zero
        :param exponential_transparency: Flag to use exponential-based decay of transparency lower than segm_threshold
        """
        super().__init__()
        n_out = 16

        norm_layer = get_norm_layer(normalization)

        self.ntex_preproc = nn.Sequential(
            EqualConv2d(in_channels_ntex, in_channels_ntex, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(in_channels_ntex, affine=True),
            FusedLeakyReLU(in_channels_ntex),
            EqualConv2d(in_channels_ntex, in_channels_ntex, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(in_channels_ntex, affine=True),
            FusedLeakyReLU(in_channels_ntex),
        )

        self.uv_preproc = nn.Sequential(
            EqualConv2d(2, 8, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(8, affine=True),
            FusedLeakyReLU(8),
            EqualConv2d(8, 8, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(8, affine=True),
            FusedLeakyReLU(8),
        )

        self.uv_preproc = nn.Sequential(
            EqualConv2d(2, 8, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(8, affine=True),
            FusedLeakyReLU(8),
            EqualConv2d(8, 8, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(8, affine=True),
            FusedLeakyReLU(8),
        )

        self.uv_mask_preproc = nn.Sequential(
            EqualConv2d(1, 4, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(4, affine=True),
            FusedLeakyReLU(4),
            EqualConv2d(4, 4, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(4, affine=True),
            FusedLeakyReLU(4),
        )

        self.model = GlobalGeneratorSkip(
            in_channels_ntex + 8 + 4,
            n_out,
            ngf=ngf,
            n_downsampling=n_downsampling,
            n_blocks=n_blocks,
            norm_layer=nn.BatchNorm2d,
            replicate_pad=replicate_pad,
        )

        self.rgb_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, 3, 3, 1, 1, bias=True, replicate_pad=replicate_pad),
            nn.Tanh())

        self.segm_head = nn.Sequential(
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, n_out, 3, 1, 1, bias=False, replicate_pad=replicate_pad),
            norm_layer(n_out, affine=True),
            FusedLeakyReLU(n_out),
            EqualConv2d(n_out, 3, 3, 1, 1, bias=True, replicate_pad=replicate_pad),
            nn.Sigmoid())

        self.segm_channels = segm_channels
        self.hardsegm = hardsegm
        self.normalization = normalization
        self.segm_threshold = segm_threshold
        self.exponential_transparency = exponential_transparency

        self.i = 0

    def forward(self, data_dict):
        """
        Get RGB and segmentation mask using rasterized SMPL-X with UV and neural texture.

        :param data_dict: Data_dict with UV-render, UV-mask and neural texture
        :return: Data dict with rendered RGB avatar and segmentation mask
        """
        uv = data_dict['uv']
        uv_mask = data_dict['uv_mask']
        tex = data_dict['ntexture']
        tex = self.ntex_preproc(tex)

        nrender = torch.nn.functional.grid_sample(tex, uv.permute(0, 2, 3, 1), align_corners=True)
        nrender = nrender * uv_mask

        uv_proc = self.uv_preproc(uv)
        uv_mask_proc = self.uv_mask_preproc(uv_mask)
        inp = torch.cat([nrender, uv_proc, uv_mask_proc], dim=1)
        out = self.model(inp)

        rgb = self.rgb_head(out)
        segm = self.segm_head(out)

        segm = segm[:, :self.segm_channels]
        segm_fg = segm[:, :1]

        if self.hardsegm:
            if 'uv_mask' in data_dict:
                uvmask = data_dict['uv_mask']
            else:
                uv = data_dict['uv']
                uvmask = ((uv > -10).sum(dim=1, keepdim=True) > 0).float()
            segm_fg = (segm_fg + uvmask).clamp(0., 1.)

        src_rgb = to_sigm(rgb)
        if 'background' in data_dict:
            background = data_dict['background']
            if self.segm_threshold != 0:
                if self.exponential_transparency:
                    # decrease
                    speed = 20
                    multiplier = 1 - torch.exp((self.segm_threshold - segm_fg) * speed)

                    # increase
                    speed = 70
                    multiplier *= 1 + torch.exp((segm_fg - self.segm_threshold - 0.2) * speed)

                    segm_fg = multiplier.clip(0, 1)
                else:
                    segm_fg[segm_fg < self.segm_threshold] = 0

            rgb_segm = src_rgb * segm_fg + background * (1. - segm_fg)
        else:
            rgb_segm = src_rgb * segm_fg
        rgb_segm = to_tanh(rgb_segm)

        out_dict = dict(fake_rgb=rgb_segm, fake_segm=segm_fg)
        return out_dict
