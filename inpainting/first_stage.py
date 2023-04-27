"""
Class for Generate + Encode and Decode + Render a neural texture with VQGAN.
"""

import cv2
import numpy as np
import torch
from rasterizers.color_rasterizer import ColorsRasterizer
from torch import nn

from utils.general_utils import instantiate_from_config, requires_grad
from utils.image_utils import sample_values_by_uv
from utils.merging import merge_textures
from utils.rasterizer import calc_normals, erode_occlusions


class NeuralRenderer(nn.Module):
    """
    The whole pipeline except for inpainting stage
    """
    def __init__(
            self,
            rasterizer_config,
            renderer_config,
            encoder_config=None,
            generator_config=None,
            compress_branch_config=None,
            ckpt_path=None,
            ignore_keys=[],
            test_phase=False,
    ):
        """
        Instantiate and load weights for Encoder, Generator, VQGAN and Renderer

        :param rasterizer_config: Config to instantiate the rasterizer
        :param renderer_config: Config to instantiate the renderer
        :param encoder_config: Config to instantiate the encoder
        :param generator_config: Config to instantiate the generator
        :param compress_branch_config: Config to instantiate VQGAN
        :param ckpt_path: Path to the checkpoint to load weights from
        :param ignore_keys: Not load weights for listed keys
        :param test_phase: Run pipeline in test mode
        """
        super().__init__()
        self.encoder = None
        self.generator = None
        self.compress_branch = None

        if encoder_config is not None:
            self.encoder = instantiate_from_config(encoder_config)
            self.encoder.eval()
            requires_grad(self.encoder, False)
        if generator_config is not None:
            self.generator = instantiate_from_config(generator_config)
            self.generator.eval()
            requires_grad(self.generator, False)
        if compress_branch_config is not None:
            self.compress_branch = instantiate_from_config(compress_branch_config)
            self.compress_branch.eval()
            requires_grad(self.compress_branch, False)

        self.rasterizer = instantiate_from_config(rasterizer_config)
        self.renderer = instantiate_from_config(renderer_config)

        self.rasterizer.eval()
        self.renderer.eval()

        requires_grad(self.rasterizer, False)
        requires_grad(self.renderer, False)

        self.normal_rasterizer = ColorsRasterizer(
            rasterizer_config.params.H,
            rasterizer_config.params.W,
            rasterizer_config.params.faces_path,
        )
        self.normal_rasterizer.eval()
        requires_grad(self.normal_rasterizer, False)

        self.test_phase = test_phase

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=None):
        """
        Load weights from the checkpoint

        :param path: Path to the checkpoint file
        :param ignore_keys: Not load weights for listed keys
        :return:
        """
        if ignore_keys is None:
            ignore_keys = list()

        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'vq_branch' in k:
                sd[k.replace('vq_branch', 'compress_branch')] = sd[k]

            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _render_normal_map(self, data_dict):
        """
        Render normals for SMPL-X models

        :param data_dict: Data dict with SMPL-X mesh set as vertices array
        :return: Data dict with rendered normals
        """
        verts = data_dict['verts']
        K = data_dict['calibration_matrix']
        with torch.no_grad():
            faces = self.normal_rasterizer.faces.long()
            verts_normals = calc_normals(verts, K, faces)
            normal_map = self.normal_rasterizer(verts, verts_normals)

        render_mask = (normal_map != 0).sum(dim=1, keepdim=True) > 0

        camera_vector = torch.from_numpy(np.array([0, 0, 1])).to(normal_map.device)
        camera_vector = camera_vector.reshape([1, 3, 1, 1])
        normal_map = normal_map * camera_vector
        normal_map = normal_map * render_mask
        normal_map = torch.sum(normal_map, 1, keepdim=True)
        normal_map = torch.clip(normal_map, 0, 1)

        data_dict['normal_map'] = normal_map
        return data_dict

    @torch.no_grad()
    def encode(self, data_dict, concat_with=None):
        """
        Encode image -> Generate Texture -> Compress texture with VQGAN

        :param data_dict: Data dict with data from a dataloader
        :param concat_with: Neural texture to concatenate with the generated one
        :return: Data dict with compressed texture
        """
        if 'ntexture' not in data_dict:
            data_dict.update(self.encoder(data_dict))
            data_dict.update(self.generator(data_dict))
            data_dict.update(self.rasterizer(data_dict))

            # Sample features from pixel-align feature extractor
            uv = data_dict['uv']
            segm = data_dict['real_segm']
            rgb = data_dict['input_rgb']
            ntex = data_dict['ntexture']

            if 'colored_uv_map' in data_dict:
                uv = erode_occlusions(uv, data_dict["colored_uv_map"])

            rgb = rgb.detach().cpu().numpy()
            rgb = (rgb + 1) * 128
            rgb = rgb.astype(np.uint8)
            pixel_proj, mask, inpaint_mask = sample_values_by_uv(uv, rgb, ntex.shape, segm)

            pixel_proj = pixel_proj.astype(np.float32)
            pixel_proj = pixel_proj / 255
            pixel_proj = torch.from_numpy(pixel_proj).to(uv.device)
            pixel_sampled_tex = pixel_proj

            mask = torch.from_numpy(mask).to(uv.device)
            inpaint_mask = torch.from_numpy(inpaint_mask).to(uv.device)

            pixel_sampled_tex = torch.cat([pixel_sampled_tex, mask, inpaint_mask], dim=1)
            data_dict['ntexture'] = torch.cat([data_dict['ntexture'], pixel_sampled_tex], dim=1)

            data_dict.update(self._render_normal_map(data_dict))
            normal_map = data_dict['normal_map'].detach().cpu().numpy()
            normal_map = normal_map * 255
            normal_map = normal_map.astype(np.uint8)
            pixel_proj, _, _ = sample_values_by_uv(uv, normal_map, ntex.shape, segm)
            pixel_proj = pixel_proj.astype(np.float32)
            pixel_proj = pixel_proj / 255
            data_dict['normal_angles'] = torch.from_numpy(pixel_proj).to(uv.device)

        if concat_with is not None:
            data_dict = merge_textures(data_dict, concat_with)

        data_dict['src_ntexture'] = data_dict['ntexture'].detach()
        data_dict['texture_mask'] = data_dict['ntexture'][:, -2:-1]
        data_dict.update(self.compress_branch(data_dict, compress_only=True))
        return data_dict

    @torch.no_grad()
    def decode(self, data_dict, copy_input_part=False):
        """
        Decompress texture with VQGAN -> Render avatar
        :param data_dict: Data dict with a compressed texture [optionally inpainted]
        :param copy_input_part: Copy known part to the inpainted one
        :return: Data dict with rendered avatar and decompressed texture
        """
        self.compress_branch.eval()
        self.renderer.eval()

        decoded = self.compress_branch.decode(data_dict['compressed'])

        if copy_input_part:
            mask = data_dict['src_ntexture'][:, -2:-1].detach()

            if self.test_phase:
                if 'binary_uv_map' in data_dict:
                    mask = mask + (1 - data_dict['binary_uv_map'])
                    mask = mask.clip(0, 1)

                mask = mask.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                kernel = np.ones((3, 3), 'uint8')

                for i in range(len(mask)):
                    mask[i, 0] = cv2.erode(mask[i, 0], kernel, iterations=1)
                    mask[i, 0] = cv2.blur(mask[i, 0], (3, 3))

                mask = torch.FloatTensor(mask).to(decoded.device)
                mask /= 255
                if 'binary_uv_map' in data_dict:
                    mask = mask - (1 - data_dict['binary_uv_map'])
                    mask = mask.clip(0, 1)

            data_dict['diffusion_ntexture'] = decoded
            data_dict['mask'] = mask
            data_dict['ntexture'] = (1 - mask) * decoded + mask * data_dict['src_ntexture']
        else:
            data_dict['ntexture'] = decoded

        data_dict.update(self.rasterizer(data_dict))
        data_dict.update(self.renderer(data_dict))
        return data_dict

    def forward(self, data_dict):
        data_dict.update(self.encode(data_dict))
        data_dict.update(self.decode(data_dict))
        return data_dict
