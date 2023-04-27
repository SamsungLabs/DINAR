"""
Class and functions to save intermediate results of inference and finetuning
"""

import os
import pickle

import cv2
import numpy as np
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from utils.general_utils import detach_dict


def data_dict_log_local_by_angle(
        save_dir,
        split,
        data_dict,
        save_all_keys=False,
):
    """
    Save certain data_dict values in the target folder.

    :param save_dir: Path to directory to save the data
    :param split: Name of subfolder (e.g Train, Test)
    :param data_dict: Data_dict with values to save
    :param save_all_keys: Flag to save all keys in the data_dict for debug purpose
    :return:
    """
    root = os.path.join(save_dir, "textures", split)
    batch_size = len(data_dict['ntexture'])
    for num in range(batch_size):
        if save_all_keys:
            sub_dict = data_dict
        else:
            keys_to_save = [
                'ntexture',
                'normal_angles',
                'view',
                'calibration_matrix',
                'verts',
                'person_id',
                'fake_rgb',
                'fake_segm',
            ]
            if 'diffusion_ntexture' in data_dict:
                keys_to_save += ['diffusion_ntexture']
            if 'mask' in data_dict:
                keys_to_save += ['mask']

            sub_dict = {k: v[num:num+1] for k, v in data_dict.items() if k in keys_to_save}

        detach_dict(sub_dict, to_numpy=True)
        sub_dict['fake_rgb'] = ((sub_dict['fake_rgb'] + 1) / 2 * 255).astype(np.uint8)
        sub_dict['fake_segm'] = (sub_dict['fake_segm'] * 255).astype(np.uint8)

        if 'smplx_param' in data_dict:
            sub_dict['smplx_param'] = {k: v[num] for k, v in data_dict['smplx_param'].items()}
            detach_dict(sub_dict['smplx_param'], to_numpy=True)

        filename = "{}.pkl".format(sub_dict['person_id'][0])

        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(sub_dict, handle)


class ImageLogger(Callback):
    """
    Callback implementation for logging results during inference and finetuning
    """
    def __init__(self, batch_frequency=10):
        """
        Initialize callback and set how often save logs

        :param batch_frequency:
        """
        super().__init__()
        self.batch_frequency = batch_frequency

    @rank_zero_only
    def image_log_local(
            self,
            save_dir,
            split,
            images,
            global_step,
            current_epoch,
            batch_idx,
            dataloader_idx,
            normalize=True,
    ):
        """
        Save images in selected folder as concatenated grid

        :param save_dir: Path to folder to save images
        :param split: Name of subfolder (e.g Train, Test)
        :param images: List of images to save
        :param global_step: Number of global step of the optimization process
        :param current_epoch: Current epoch number
        :param batch_idx: Index of current batch
        :param dataloader_idx: Index of used dataloader
        :param normalize: Flag renormalize images -1,1 -> 0,1
        :return:
        """
        root = os.path.join(save_dir, "images", split)
        grid = torchvision.utils.make_grid(images, nrow=4)

        if normalize:
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.clamp(0, 1)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.detach().cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = "{:06}_e-{:06}_b-{:06}-loader={:06}.png".format(
            global_step,
            current_epoch,
            batch_idx,
            dataloader_idx,
        )
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        cv2.imwrite(path, grid[..., ::-1])

    def _log_data(self, stage, pl_module, batch, batch_idx, dataloader_idx, save_all_keys=False, dataset_name=None):
        """
        Log data provided with batch in accordance with stage

        :param stage: ["train", "test", "val"]
        :param pl_module: Model to process batch data
        :param batch: Data stored as a batch from a dataloader
        :param batch_idx: Index of a batch
        :param dataloader_idx: Index of a dataloader
        :param save_all_keys: Flag to save all keys in the batch's data dicts
        :param dataset_name: Name of used dataset
        :return:
        """
        real_rgb = None
        real_segm = None
        if 'real_rgb' in batch:
            real_rgb = batch['real_rgb']
        if 'real_segm' in batch:
            real_segm = batch['real_segm']

        fake_rgb = batch['fake_rgb']
        fake_segm = batch['fake_segm']
        textures = batch['ntexture'][:, -5:-2]

        def get_make_path_fn(dataset_name):
            dataset_name = f'/{dataset_name}' if dataset_name else ''
            def _inner(folder_name):
                return "/".join([dataset_name, folder_name])
            return _inner
        make_path = get_make_path_fn(dataset_name)

        real_rgb_path = make_path("real_rgb")
        fake_rgb_path = make_path("fake_rgb")
        real_segm_path = make_path("real_segm")
        fake_segm_path = make_path("fake_segm")
        if real_rgb is not None:
            self.image_log_local(pl_module.logger.save_dir, stage + real_rgb_path, real_rgb,
                                 pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx)
        self.image_log_local(pl_module.logger.save_dir, stage + fake_rgb_path, fake_rgb,
                             pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx)

        if real_segm is not None:
            self.image_log_local(pl_module.logger.save_dir, stage + real_segm_path, real_segm,
                             pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx,
                             normalize=False)
        self.image_log_local(pl_module.logger.save_dir, stage + fake_segm_path, fake_segm,
                             pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx,
                             normalize=False)

        textures_path = make_path("textures")
        self.image_log_local(pl_module.logger.save_dir, stage + textures_path, textures,
                             pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx,
                             normalize=False)

        if stage == 'test':
            data_dict_log_local_by_angle(pl_module.logger.save_dir, "data_dict", batch, save_all_keys)

        elif 'fake_rotated_rgb' in batch:
            fake_rotated_rgb = batch['fake_rotated_rgb']
            fake_rotated_path = make_path("fake_rotated_rgb")
            self.image_log_local(pl_module.logger.save_dir, stage + fake_rotated_path, fake_rotated_rgb,
                                 pl_module.global_step, pl_module.current_epoch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.is_global_zero and batch_idx % self.batch_frequency == 0:
            if isinstance(outputs, list):
                outputs = outputs[0]
            self._log_data('train', pl_module, outputs['data_dict'], batch_idx, dataloader_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.is_global_zero:
            for dataset_name, data_dict in outputs['data_dicts'].items():
                self._log_data('val', pl_module, data_dict, batch_idx, dataloader_idx, dataset_name=dataset_name)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if trainer.is_global_zero:
            self._log_data('test', pl_module, outputs['data_dict'], batch_idx, dataloader_idx)
