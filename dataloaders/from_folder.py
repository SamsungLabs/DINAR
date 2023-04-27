import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch.utils.data

from utils.dataloader_utils import get_ltrb_from_verts_and_adjust, update_smplifyx_after_crop_and_resize, \
    get_ltrb_from_segm_and_adjust, combine_ltrb, crop_img_ltrb, upd_cropmask, load_colored_uv_map
from utils.general_utils import to_tanh, make_K
from utils.image_utils import itt, tti, rgb2tensor, segm2tensor
from utils.smplx_model import build_smplx_model_dict, pass_smplx_dict


class FromFolder(torch.utils.data.Dataset):
    """
    Dataloader for data provided as folder with RGB, Segmentation and SMPL-X fits subfolders.
    Alternative to loaders of picklified data
    """
    def __init__(
            self,
            data_root,
            v_inds_path,
            smplx_path,
            image_size=512,
            scale_bbox=1.2,
            rgb_folder="rgb",
            segm_folder="segm",
            smplx_folder="smplx",
            colored_uv_map=None,
            binary_uv_map=None,
            additive_uv_map=None,
            repeat_data=None,
            frames_subset=None,
    ):
        """
        Initialize dataloader. Find data in provided folders.

        :param data_root: Root folders with all dataset's meta data
        :param v_inds_path: Path to numpy file with SMPL-X indices sublist in fixed order
        :param smplx_path: Path to SMPL-X models
        :param image_size: Target rendering size
        :param scale_bbox: Bounding box rescale factor
        :param rgb_folder: Subfolder with RGB images
        :param segm_folder: Subfolder with Segmentation images
        :param smplx_folder: Subfolder with SMPL-X fits
        :param colored_uv_map: Color encoded uv map for self occlusion detection
        :param binary_uv_map: Binary mask for uv map to erode only inner edges
        :param additive_uv_map: Binary mask to add on top of the calculated one
        :param repeat_data: Number of iteration throw data in one epoch
        :param frames_subset: Subset of frames to process
        """
        self.rgb_folder = os.path.join(data_root, rgb_folder)
        self.segm_folder = os.path.join(data_root, segm_folder)
        self.smplx_folder = os.path.join(data_root, smplx_folder)

        self.rgb_files = glob(os.path.join(self.rgb_folder, '*.*'))
        if frames_subset is not None:
            self.rgb_files = [fn for fn in self.rgb_files if fn.split("/")[-1].split(".")[0] in frames_subset]
        if repeat_data is not None:
            self.rgb_files = self.rgb_files * repeat_data

        self._len = len(self.rgb_files)

        self.v_inds = np.load(v_inds_path)

        self.smplx_model_dict = build_smplx_model_dict(smplx_path, device='cpu')
        self.image_size = image_size
        self.scale_bbox = scale_bbox

        self.colored_uv_map = load_colored_uv_map(colored_uv_map)
        self.binary_uv_map = load_colored_uv_map(binary_uv_map, binary=True)
        self.additive_uv_map = load_colored_uv_map(additive_uv_map, binary=True)

        self.data_hash = {}

    def load_smplifyx(self, smplx_filename, rgb, segm):
        """
        Load SMPL-X parameters and generate SMPL-X mesh.
        
        :param smplx_filename: Path to pickle with SMPL-X parameters
        :param rgb: RGB image corresponding to SMPL-X fit
        :param segm: Segmentation map corresponding to SMPL-X fit
        :return: Data dict with inferred SMPL-X vertices; adjusted RGB and segmentation masks; Bounding box
        """
        with open(smplx_filename, 'rb') as f:
            smplx_params_dict = pickle.load(f)
            if 'result' in smplx_params_dict:
                smplx_params_dict = smplx_params_dict['result']
            if 'smplx' in smplx_params_dict:
                smplx_params_dict = smplx_params_dict['smplx']

        if 'gender' not in smplx_params_dict:
            smplx_params_dict['gender'] = 'neutral'
        if 'camera_translation' in smplx_params_dict:
            smplx_params_dict['transl'] = smplx_params_dict.pop('camera_translation')
        smplx_params_dict.pop('frame_fns', None)
        smplx_params_dict.pop('camera_rotation', None)
        smplx_params_dict['reye_pose'] = np.array([[-0.39876276, -0.0913531,  -0.07875713]])
        smplx_params_dict['leye_pose'] = np.array([[-0.23671481, -0.14406434,  0.10868622]])
        
        smplx_output = pass_smplx_dict(smplx_params_dict, self.smplx_model_dict, 'cpu')
        verts = smplx_output['vertices']

        if 'ltrb' in smplx_params_dict:
            image_ltrb = smplx_params_dict['ltrb']
            rgb = crop_img_ltrb(rgb, image_ltrb)
            segm = crop_img_ltrb(segm, image_ltrb)
        else:
            image_ltrb = []
        H, W, _ = rgb.shape
        K = make_K(H, W)

        gender = smplx_params_dict['gender']
        verts = verts @ K.T
        verts = verts[self.v_inds]

        joints = smplx_output['joints']
        joints = joints @ K.T

        return dict(
            verts=verts,
            calibration_matrix=K,
            joints=joints,
            gender=gender,
            smplx_param=smplx_params_dict,
        ), rgb, segm, image_ltrb

    def load_sample(self, pid):
        """
        Load sample of data for selected person ID

        :param pid: Person ID
        :return: Loaded sample
        """
        # Read RGB and get camera parameters
        rgb_filename = os.path.join(self.rgb_folder, pid+'.png')
        rgb = cv2.imread(str(rgb_filename))[..., ::-1]

        # Read segmentation
        segm_filename = os.path.join(self.segm_folder, pid+'.png')
        segm = cv2.imread(str(segm_filename))[..., ::-1]

        # Read SMPLX model
        smplx_filename = os.path.join(self.smplx_folder, pid + '.pkl')
        smplifyx, rgb, segm, image_ltrb = self.load_smplifyx(smplx_filename, rgb, segm)
        smplx_param = smplifyx['smplx_param']

        # Calculate bounding box
        H, W, _ = rgb.shape
        ltrb_verts = get_ltrb_from_verts_and_adjust(smplifyx['verts'], self.scale_bbox)
        ltrb_segm = get_ltrb_from_segm_and_adjust(segm[..., 0], self.scale_bbox, H, W)
        ltrb = combine_ltrb(ltrb_verts, ltrb_segm)

        # Crop and resize rgb and segmentation
        segmentation_pt = segm2tensor(segm, ltrb, self.image_size)
        rgb_pt, cm_pt, cropped_sizes = rgb2tensor(rgb, ltrb, self.image_size)

        segmentation_pt = (segmentation_pt > 0.2).float() * 1

        # Rescale vertices
        smplifyx = update_smplifyx_after_crop_and_resize(
            smplifyx,
            ltrb,
            cropped_sizes,
            (self.image_size, self.image_size),
        )
        verts = torch.FloatTensor(smplifyx['verts'])
        K = torch.FloatTensor(smplifyx['calibration_matrix'])
        cropmask = upd_cropmask(tti(cm_pt), verts.numpy())
        cm_pt = itt(cropmask)

        background = torch.ones(3, rgb_pt.shape[1], rgb_pt.shape[2])
        rgb_pt_with_bg = rgb_pt * segmentation_pt + (1. - segmentation_pt) * background
        rgb_pt_with_bg = to_tanh(rgb_pt_with_bg)

        rgb_pt = rgb_pt * segmentation_pt
        rgb_pt = to_tanh(rgb_pt)

        segmentation_pt[segmentation_pt < 0.1] = 0

        # Store everything into data_dict
        sample = {
            'person_id': pid,
            'verts': verts,
            'calibration_matrix': K,
            'crop_mask': cm_pt,
            'input_rgb': rgb_pt,
            'real_rgb': rgb_pt_with_bg,
            'background': background,
            'real_segm': segmentation_pt,
            'smplx_param': smplx_param,
        }
        if self.colored_uv_map is not None:
            sample['colored_uv_map'] = self.colored_uv_map
        if self.binary_uv_map is not None:
            sample['binary_uv_map'] = self.binary_uv_map
        if self.additive_uv_map is not None:
            sample['additive_uv_map'] = self.additive_uv_map
        return sample

    def __getitem__(self, index):
        index = index % self._len
        rgb_filename = self.rgb_files[index]
        pid = rgb_filename.split('/')[-1].split('.')[0]

        if rgb_filename in self.data_hash:
            data_dict = self.data_hash[rgb_filename]
        else:
            data_dict = self.load_sample(pid)
            self.data_hash[rgb_filename] = data_dict
        return data_dict

    def __len__(self):
        return self._len
