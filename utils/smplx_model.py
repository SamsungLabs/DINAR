import os

import torch

from utils import smplx_flame


def build_smplx_model_dict(smplx_model_dir, device, use_pca=True, num_pca_comps=6):
    """
    Build dict with MALE, FEMALE and NEUTRAL parametric models.

    :param smplx_model_dir: Folder with parametric models parameters
    :param device: Device to instantiate SMPL-X models on
    :param use_pca: Flag to use PCA for hands
    :param num_pca_comps: Number of PCA components for hands
    :return: Dict with SMPL-X models
    """
    gender2filename = dict(neutral='SMPLX_NEUTRAL.pkl', male='SMPLX_MALE.pkl', female='SMPLX_FEMALE.pkl')
    gender2path = {k: os.path.join(smplx_model_dir, v) for (k, v) in gender2filename.items()}
    gender2model = {k: smplx_flame.SMPLX(
        v,
        use_pca=use_pca,
        use_face_contour=True,
        num_pca_comps=num_pca_comps,
    ).to(device) for (k, v) in gender2path.items()}

    return gender2model


def pass_smplx_dict(smplx_params_dict, smplx_model_dict, device):
    """
    Calculate vertices and joints positions by SMPL-X pose and shape vectors

    :param smplx_params_dict: Dict of SMPL-X parameters (shape, pose)
    :param smplx_model_dict:  Dict of SMPL-X models (for three genders)
    :param device: Target device for inference
    :return: Dict with resulting vertices and joints positions
    """
    gender = smplx_params_dict['gender']
    smplx_input = {}

    for k, v in smplx_params_dict.items():
        if k != 'gender':
            if type(v) != torch.Tensor:
                smplx_input[k] = torch.FloatTensor(v).to(device)
            else:
                smplx_input[k] = v.to(device)

    smplx_input['right_hand_pose'] = smplx_input['right_hand_pose'][:, :6]
    smplx_input['left_hand_pose'] = smplx_input['left_hand_pose'][:, :6]
    with torch.no_grad():
        smplx_output = smplx_model_dict[gender](**smplx_input)
    vertices = smplx_output.vertices.cpu().numpy()[0]
    joints = smplx_output.joints.cpu().numpy()[0]
    smplx_output_dict = dict(vertices=vertices, joints=joints)
    return smplx_output_dict
