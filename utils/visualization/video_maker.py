"""
Implementation of a class for video generation.
"""

import math
import os
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.dataloader_utils import rotate_verts
from utils.general_utils import instantiate_from_config, dict2device
from utils.image_utils import tti, itt
from utils.smplx_model import pass_smplx_dict, build_smplx_model_dict
from utils.visualization.av_utils import VideoWriter


class VideoMaker:
    """
    Make video with provided neural avatar
    """
    def __init__(
            self,
            renderer_config,
            rasterizer_config,
            ckpt_path,
            smplx_path,
            v_inds_path,
    ):
        """
        Instantiate renderer and rasterizer. Load models weights from the checkpoint.
        Make smplx model.

        :param renderer_config: OmegaConf config with renderer parameters
        :param rasterizer_config: OmegaConf config with rasterizer parameters
        :param ckpt_path: Pretrained checkpoint with renderer and rasterizer weights
        :param smplx_path: Path to the directory with SMPL-X parameters
        :param v_inds_path: Path to a file with vertices indices in the target order
        """
        self.renderer = instantiate_from_config(renderer_config)
        self.rasterizer = instantiate_from_config(rasterizer_config)

        state_dict = torch.load(ckpt_path)
        for key in list(state_dict['state_dict'].keys()):
            if key.startswith('first_stage.'):
                state_dict['state_dict'][key[len('first_stage.'):]] = state_dict['state_dict'].pop(key)

        self.renderer.load_state_dict(self._load_module(state_dict, 'renderer'))
        self.rasterizer.load_state_dict(self._load_module(state_dict, 'rasterizer'))

        self.renderer.eval()
        self.rasterizer.eval()

        self.renderer.cuda()
        self.rasterizer.cuda()

        self.smplx_model_dict = build_smplx_model_dict(smplx_path, device='cuda')
        self.v_inds = np.load(v_inds_path)

    def _load_module(self, state_dict, module_name):
        """
        Select submodule from a state dict by name

        :param state_dict: Loaded state dict (checkpoint with weights)
        :param module_name: Name of a module to extract weights for
        :return: State dict for selected module only
        """
        return {k.replace(module_name + '.', ''): v
                for k, v in state_dict['state_dict'].items()
                if k.startswith(module_name)}

    def _dict_to_tensor(self, data_dict):
        """
        Transform all ndarrays from the dict into tensors.
        Data storied in pickles as ndarray because it more compact.

        :param data_dict: Dict with data to transform
        :return: Pointer to the same dict with ndarrays converted into tensors
        """
        for key in data_dict.keys():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.FloatTensor(data_dict[key])

    def write_seq_animated_video(
            self,
            data_dict,
            video_file,
            path_to_sequence,
            fps=30,
            zero_threshold=0.5,
    ):
        """
        Write a video with animation sequence

        :param data_dict: Loaded from a file data_dict
        :param video_file: Target file to save the video
        :param path_to_sequence: Path to the animation sequence. Sequence represented as smpl-x parameters for each frame
        :param fps: Frame per second
        :param zero_threshold: Threshold for segmentation mask to separate an avatar from the background
        :return:
        """
        self.renderer.segm_threshold = zero_threshold
        images_folder = os.path.splitext(video_file)[0]
        segms_folder = images_folder + '_segm'
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        if not os.path.exists(segms_folder):
            os.makedirs(segms_folder)

        vw = VideoWriter(video_file, fps)
        self._dict_to_tensor(data_dict)
        data_dict = dict2device(data_dict, 'cuda')

        with open(path_to_sequence, 'rb') as f:
            animation_dict = pickle.load(f)

        smplx_dict = data_dict['smplx_param']
        gender = smplx_dict['gender']
        smplx_dict['reye_pose'] = np.array([[-0.39876276, -0.0913531, -0.07875713]])
        smplx_dict['leye_pose'] = np.array([[-0.23671481, -0.14406434, 0.10868622]])
        smplx_dict = {k: torch.FloatTensor(v).cuda().to(torch.float32) for k, v in smplx_dict.items() if k != 'gender'}
        smplx_dict['gender'] = gender
        K = data_dict['calibration_matrix'][0]

        background = torch.ones(1, 3, self.rasterizer.H, self.rasterizer.W)
        background = torch.clip(background, 0, 1).to('cuda')

        data_dict['background'] = background

        data_dict['ntexture_quant'] = data_dict['ntexture']

        animation_length = len(animation_dict['body_params_list'])
        print('Animation length:', animation_length)
        for frame_num in tqdm(range(animation_length)):
            smplx_dict['body_pose'] = animation_dict['body_params_list'][frame_num]['body_pose']
            smplx_dict['global_orient'] = animation_dict['body_params_list'][frame_num]['global_orient']
            smplx_dict['right_hand_pose'] = animation_dict['body_params_list'][frame_num]['right_hand_pose']
            smplx_dict['left_hand_pose'] = animation_dict['body_params_list'][frame_num]['left_hand_pose']
            smplx_dict['jaw_pose'] = animation_dict['body_params_list'][frame_num]['jaw_pose']

            smplx_output = pass_smplx_dict(smplx_dict, self.smplx_model_dict, 'cuda')

            verts = smplx_output['vertices']
            if 'RT' in smplx_dict:
                RT = smplx_dict['RT'].cpu().numpy()
                verts = np.concatenate([verts, np.ones_like(verts[:, :1])], axis=1)
                verts = verts @ RT.T
            verts = torch.FloatTensor(verts).cuda()

            verts = verts @ K.T
            verts = verts[self.v_inds]

            data_dict['verts'] = torch.unsqueeze(verts, 0)
            data_dict.update(self.rasterizer(data_dict))
            data_dict.update(self.renderer(data_dict))

            rgb = data_dict['fake_rgb']
            segm = data_dict['fake_segm']

            segm = tti(segm)
            segm = itt(segm).to(rgb.device)

            frame = tti(rgb)
            frame = (frame + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            frame = (frame * 255).astype(np.uint8)
            vw.add_frame(frame)

            rgb = tti(rgb)
            rgb = (rgb + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            rgb = (rgb * 255).astype(np.uint8)

            filename = str(frame_num)
            cv2.imwrite(os.path.join(images_folder, filename + '.png'), rgb[..., ::-1])
            cv2.imwrite(os.path.join(segms_folder, filename + '.png'), segm[0].detach().cpu().numpy() * 255)

        vw.close()

    def write_rotation_video(
            self,
            data_dict,
            video_file,
            duration_sec=12,
            fps=30,
            zero_threshold=0.5,
    ):
        """
        Write a video with rotated around vertical axis avatar

        :param data_dict: Loaded from a file data_dict
        :param video_file: Target file to save the video
        :param duration_sec: Duration of the target video
        :param fps: Frame per second
        :param zero_threshold: Threshold for segmentation mask to separate an avatar from the background
        :return:
        """
        self.renderer.segm_threshold = zero_threshold
        images_folder = os.path.splitext(video_file)[0]
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)

        vw = VideoWriter(video_file, fps)
        for key in ['verts', 'calibration_matrix', 'ntexture']:
            data_dict[key] = torch.FloatTensor(data_dict[key])
        data_dict = dict2device(data_dict, 'cuda')

        verts = data_dict['verts'].clone().detach()
        K = data_dict['calibration_matrix']

        background = torch.ones(1, 3, self.rasterizer.H, self.rasterizer.W)
        background = torch.clip(background, 0, 1).to('cuda')
        data_dict['background'] = background

        data_dict['ntexture_quant'] = data_dict['ntexture']

        step = 1 / (fps * duration_sec)
        angle = 0.
        for num in tqdm(np.arange(0, 1, step)):
            data_dict.update(self.rasterizer(data_dict))
            data_dict.update(self.renderer(data_dict))

            rgb = data_dict['fake_rgb']

            frame = tti(rgb)
            frame = (frame + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            frame = (frame * 255).astype(np.uint8)
            vw.add_frame(frame)

            target_angle = 2 * np.pi * angle
            filename = str(int(math.degrees(target_angle)))
            cv2.imwrite(os.path.join(images_folder, filename + '.png'), frame[..., ::-1])

            data_dict['verts'] = rotate_verts(verts, target_angle, K)
            angle += step

        vw.close()
