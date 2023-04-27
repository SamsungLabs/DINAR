"""
Lightning module for RGB channels finetuning
"""

import pickle

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from kornia.filters import box_blur
from kornia.morphology import erosion

from pl_callbacks.image_logger import data_dict_log_local_by_angle
from utils.general_utils import dict2device, requires_grad, instantiate_from_config, disabled_train, detach_dict
from utils.image_utils import get_lossmask
from utils.zero_adam import ZeroAdam


class TextureTune(pl.LightningModule):
    """
    Class for finetuning RGB channels of the neural texture
    """
    def __init__(
            self,
            ckpt_path,
            pretrained_texture,
            renderer_config,
            discriminator_config,
            rasterizer_config,
            criteria_config,
            lr_G=1e-3,
            lr_D=1e-3,
            rescale_steps=64,
    ):
        """
        Initialize renderer and discriminator to calculate gradients for the textur

        :param ckpt_path: Path to checkpoint for loading weights
        :param pretrained_texture: Path to pickle with pretrained texture
        :param renderer_config: Config to instantiate renderer
        :param discriminator_config: Config to instantiate discriminator
        :param rasterizer_config: Config to instantiate rasterizer
        :param criteria_config: Config to instantiate loses
        :param lr_G: Learning rate of the generator
        :param lr_D: Learning rate of the discriminator
        :param rescale_steps: Number of steps for finetuning color scale and shift
        """
        super().__init__()
        self.lr_G = lr_G
        self.lr_D = lr_D

        self.renderer = instantiate_from_config(renderer_config)

        self.rasterizer = instantiate_from_config(rasterizer_config)
        self.discriminator = instantiate_from_config(discriminator_config)

        self.renderer.eval()
        self.renderer.train = disabled_train
        self.rasterizer.eval()
        self.rasterizer.train = disabled_train

        self.criteria = {}
        for criteria_name, criterion_config in criteria_config.items():
            criterion = instantiate_from_config(criterion_config)
            self.criteria[criteria_name] = criterion
        self.criteria = dict2device(self.criteria, 'cuda')

        with open(pretrained_texture, 'rb') as handle:
            data_dict = pickle.load(handle)
        data_dict['smplx_param'] = {k: np.expand_dims(v, axis=0) for k, v in data_dict['smplx_param'].items()}
        self.data_dict_to_save = data_dict

        self.tunable_texture = torch.tensor(data_dict['ntexture'][:, -5:-2], requires_grad=True)
        self.tunable_texture = torch.nn.Parameter(self.tunable_texture, requires_grad=True)
        self.known_mask = torch.tensor(data_dict['mask'])
        if 'diffusion_ntexture' in data_dict:
            self.src_texture = torch.tensor(data_dict['diffusion_ntexture'])
            self.src_rgb = torch.tensor(data_dict['diffusion_ntexture'][:, -5:-2])
        else:
            self.src_texture = torch.tensor(data_dict['ntexture'])
            self.src_rgb = torch.tensor(data_dict['ntexture'][:, -5:-2])

        self.scales = torch.ones([1, 3, 1, 1], requires_grad=True)
        self.scales = torch.nn.Parameter(self.scales, requires_grad=True)
        self.shifts = torch.zeros([1, 3, 1, 1], requires_grad=True)
        self.shifts = torch.nn.Parameter(self.shifts, requires_grad=True)
        self.rescale_steps = rescale_steps

        mask = data_dict['normal_angles']

        # Erode mask to avoid updating contours of the texture parts
        mask = np.transpose(mask, (0, 2, 3, 1))
        for i in range(mask.shape[0]):
            speed = 20
            threshold = 0.6
            mask_blurred = mask[i].copy()
            mask_blurred *= np.maximum(1 - np.exp((threshold - mask_blurred) * speed), 0)

            mask_blurred = cv2.blur(mask_blurred, (5, 5))
            mask[i] = mask_blurred[..., None]
        mask = np.transpose(mask, (0, 3, 1, 2))

        self.grad_mask = torch.FloatTensor(mask)
        self.tunable_texture.register_hook(lambda grad: grad * self.grad_mask)

        self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        """
        Load models weights from the provided chekpoints

        :param path: Path to the checkpoint
        :return:
        """
        state_dict = torch.load(path)['state_dict']
        for key in list(state_dict.keys()):
            if key.startswith('first_stage.'):
                state_dict[key[len('first_stage.'):]] = state_dict.pop(key)
        for key in list(state_dict.keys()):
            if key.split('.')[0] not in ['renderer', 'discriminator', 'rasterizer']:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    def tunable_texture_to_device(self):
        """
        Move all related tensors to the same device

        :return:
        """
        self.tunable_texture = self.tunable_texture.to(self.device)
        self.grad_mask = self.grad_mask.to(self.device)
        self.src_texture = self.src_texture.to(self.device)
        self.src_rgb = self.src_rgb.to(self.device)

    def forward(self, data_dict, mix_with_source=False):
        batch_size = len(data_dict['real_rgb'])
        src_texture = self.src_texture.clone()
        src_rgb = self.src_rgb.clone()

        data_dict['ntexture'] = src_texture.clone()
        if self.global_step >= self.rescale_steps or mix_with_source:
            mask = self.known_mask.to(self.tunable_texture.device)
            unvisable_part = src_rgb * self.scales + self.shifts
            data_dict['ntexture'][:, -5:-2] = self.tunable_texture * mask + unvisable_part * (1 - mask)
        else:
            data_dict['ntexture'][:, -5:-2] = src_rgb * self.scales + self.shifts

        if mix_with_source:
            mask = self.known_mask.to(self.tunable_texture.device)

            outer_area = 1 - data_dict['binary_uv_map']
            if 'binary_uv_map' in data_dict:
                mask = mask + outer_area
                mask = mask.clip(0, 1)
            mask = box_blur(mask, (10, 10))
            if 'binary_uv_map' in data_dict:
                mask = mask - outer_area
                mask = mask.clip(0, 1)

            if 'additive_uv_map' in data_dict:
                mask = mask + data_dict['additive_uv_map']
                mask = mask.clip(0, 1)
            src_texture[:, -5:-2] = src_rgb * self.scales + self.shifts
            data_dict['ntexture'] = data_dict['ntexture'] * mask + src_texture * (1 - mask)

        data_dict['ntexture'] = data_dict['ntexture'].repeat(batch_size, 1, 1, 1)

        data_dict.update(self.rasterizer(data_dict))

        uv_mask = data_dict['uv_mask']
        kernel = torch.ones(3, 3).to(uv_mask.device)
        uv_mask = erosion(uv_mask, kernel)

        data_dict['uv_mask'] = uv_mask
        data_dict['uv'] = data_dict['uv'] * uv_mask - torch.ones_like(data_dict['uv']) * (1 - uv_mask)

        data_dict.update(self.renderer(data_dict))

        return data_dict

    def configure_optimizers(self):
        """
        Create Generator and Discriminator optimizers

        :return:
        """
        optimizer_G = ZeroAdam(
            [self.tunable_texture, self.scales, self.shifts],
            lr=self.lr_G,
            betas=(0, 0.99),
        )

        optimizer_D = ZeroAdam(
            self.discriminator.parameters(),
            lr=self.lr_D,
            betas=(0, 0.99),
        )

        return [optimizer_G, optimizer_D], []

    def texture_mode(self):
        """
        Switch model setting to train Generator

        :return:
        """
        if self.global_step >= self.rescale_steps:
            self.tunable_texture.requires_grad = True
            self.scales.requires_grad = False
            self.shifts.requires_grad = False
        else:
            self.tunable_texture.requires_grad = False
            self.scales.requires_grad = True
            self.shifts.requires_grad = True
        requires_grad(self.discriminator, False)
        self.discriminator.eval()

    def discriminator_mode(self):
        """
        Switch model setting to train Discriminator

        :return:
        """
        self.tunable_texture.requires_grad = False
        self.scales.requires_grad = False
        self.shifts.requires_grad = False
        requires_grad(self.discriminator, True)
        self.discriminator.train()

    def texture_tuning_step(self, data_dict):
        """
        One finetuning iteration of a texture

        :param data_dict: Data dict with data from the dataloader
        :return: Loss values
        """
        self.texture_mode()
        data_dict = self.forward(data_dict)

        lossmask = get_lossmask(data_dict['uv'], data_dict['real_segm'], intersection=True)
        data_dict['disc_input'] = data_dict['fake_rgb'] * lossmask
        fake_scores = self.discriminator(data_dict)
        data_dict['fake'] = fake_scores

        loss = 0
        for criterion_name, criterion in self.criteria.items():
            if criterion_name.endswith('_reg'):
                continue
            local_loss = criterion(data_dict)
            self.log('train_loss/' + criterion_name, local_loss, sync_dist=True)
            loss += local_loss
        self.log('train_loss/loss', loss, sync_dist=True)
        return loss

    def discriminator_tuning_step(self, data_dict):
        """
        One discriminator tuning step

        :param data_dict: Data dict with data from the dataloader
        :return: Loss values
        """
        self.discriminator_mode()
        with torch.no_grad():
            data_dict = self.forward(data_dict)

        lossmask = get_lossmask(data_dict['uv'], data_dict['real_segm'], intersection=True)
        data_dict['disc_input'] = data_dict['fake_rgb'] * lossmask
        fake_scores = self.discriminator(data_dict)
        data_dict['fake'] = fake_scores

        data_dict['disc_input'] = data_dict['real_rgb'] * lossmask
        real_scores = self.discriminator(data_dict)
        data_dict['real'] = real_scores

        loss = self.criteria['adversarial_loss'](data_dict)
        self.log('discriminator_loss/loss', loss, sync_dist=True)
        return loss

    def discriminator_regularization(self, data_dict):
        """
        Regularization step for discriminator

        :param data_dict: Data dict with data from the dataloader
        :return: Loss values
        """
        self.discriminator_mode()
        data_dict['disc_input'] = data_dict['real_rgb']
        real_scores = self.discriminator(data_dict)
        data_dict['real'] = real_scores

        data_dict['disc_input'].requires_grad = True
        real_scores = self.discriminator(data_dict)
        data_dict['real'] = real_scores

        r1_loss = self.criteria['r1_reg'](data_dict)
        self.log('discriminator_loss/r1_reg', r1_loss, sync_dist=True)
        return r1_loss

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        """
        One texture finetuning step

        :param train_batch: Batch of data from a dataloader
        :param batch_idx: Index of the batch
        :param optimizer_idx: Index of the optimizer
        :return: Losses and logs
        """
        data_dict = train_batch.copy()

        opt = self.optimizers()[optimizer_idx]
        opt.lr = self.lr_G

        data_dict = data_dict.copy()
        data_dict = dict2device(data_dict, self.device)
        self.tunable_texture_to_device()

        if optimizer_idx == 0:
            # autoencoder
            loss = self.texture_tuning_step(data_dict)

        if optimizer_idx == 1:
            # discriminator
            opt.lr = self.lr_D

            if self.criteria['r1_reg'].is_called():
                loss = self.discriminator_regularization(data_dict)
            else:
                loss = self.discriminator_tuning_step(data_dict)

        self.log('monitoring_step', self.global_step, sync_dist=True)
        detach_dict(data_dict)
        return {'loss': loss, 'data_dict': dict2device(data_dict, "cpu")}

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        """
        One validation step

        :param val_batch: Batch of data from a dataloader
        :param batch_idx: Index of the batch
        :param dataloader_idx: Index of the optimizer
        :return: Losses and logs
        """
        # Assuming that there is only one val_dataloader
        dataloader_name, data_dict = list(val_batch.items())[0]
        batch_size = len(data_dict['real_rgb'])
        data_dict = dict2device(data_dict, self.device)
        self.tunable_texture_to_device()

        with torch.no_grad():
            data_dict = self.forward(data_dict, mix_with_source=True)

        loss = 0
        for criterion_name, criterion in self.criteria.items():
            if criterion_name.endswith('_reg') or criterion_name == 'adversarial_loss':
                continue
            local_loss = criterion(data_dict)
            self.log('val_loss/' + criterion_name, local_loss, batch_size=batch_size, sync_dist=True)
            loss += local_loss
        self.log('val_loss/loss', loss, batch_size=batch_size, sync_dist=True)

        if batch_idx == 0:
            self.data_dict_to_save['ntexture'] = data_dict['ntexture'][0:1].detach().cpu().numpy()
            source_person_id = self.data_dict_to_save['person_id'][0]
            log_meta = f"s{self.global_step:07}_{self.data_dict_to_save['person_id'][0]}"

            self.data_dict_to_save['person_id'][0] = log_meta
            data_dict_log_local_by_angle(
                self.logger.save_dir,
                "tuned_data_dict",
                self.data_dict_to_save,
                save_all_keys=False,
            )
            self.data_dict_to_save['person_id'][0] = source_person_id

        return {"data_dicts": {dataloader_name: dict2device(data_dict, "cpu")}}
