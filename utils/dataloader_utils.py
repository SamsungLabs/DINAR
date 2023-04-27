"""
Functions to preprocess a data in dataloaders.
Mostly functions to work with bounding boxex.
"""
import cv2
import numpy as np
import torch

from utils.image_utils import get_square_bbox, scale_bbox, crop_img


def crop_img_ltrb(image, ltrb):
    """
    Crop an image by bounding box coordinates

    :param image: Input image
    :param ltrb: (left, top, right, bottom) coordinates of the bounding box
    :return: Cropped copy of the image
    """
    l, t, r, b = ltrb
    w = r - l
    h = b - t
    image = crop_img(image, t, l, h, w)
    return image


def get_ltrb_from_verts(verts):
    """
    Calculate (left, top, right, bottom) coordinates of bounding box around projected vertices

    :param verts: List of vertices in the homogeneous coordinates
    :return: (left, top, right, bottom)
    """
    verts_projected = (verts / (verts[:, 2:]))[:, :2]

    x = verts_projected[:, 0]
    y = verts_projected[:, 1]

    # get bbox in format (left, top, right, bottom)
    l = int(np.min(x))
    t = int(np.min(y))
    r = int(np.max(x))
    b = int(np.max(y))

    return (l, t, r, b)


def get_ltrb_from_verts_and_adjust(verts, scale_bbox_value):
    """
    Calculate (left, top, right, bottom) by vertices coordinates and scale by scale_bbox_value

    :param verts: List of vertices in the homogeneous coordinates
    :param scale_bbox_value: Scale factor of bounding box
    :return: Calculated, squared and scaled bounding box as (left, top, right, bottom)
    """
    ltrb = get_ltrb_from_verts(verts)
    ltrb = get_square_bbox(ltrb)
    ltrb = scale_bbox(ltrb, scale_bbox_value)

    return ltrb


def update_smplifyx_after_crop_and_resize(smplx, bbox, image_shape, new_image_shape):
    """
    Update vertices positions and calibration matrix parameters according to bounding box

    :param smplx: Dict with SMPL-X parameters
    :param bbox: Bounding box coordinates
    :param image_shape: Source image shape
    :param new_image_shape: Target image shape
    :return: Dict with upodated vertices and new calibration matrix
    """
    # it's supposed that it smplifyx's verts are in trivial camera coordinates
    fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0
    # crop
    cx, cy = cx - bbox[0], cy - bbox[1]
    # scale
    h, w = image_shape
    new_h, new_w = new_image_shape

    h_scale, w_scale = new_w / w, new_h / h

    fx, fy = fx * w_scale, fy * h_scale
    cx, cy = cx * w_scale, cy * h_scale

    # update verts
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])

    return {
        'verts': smplx['verts'] @ K.T,
        'calibration_matrix': K @ smplx['calibration_matrix'],
    }


def get_ltrb_from_segm(segm):
    """
    Calculate (left, top, right, bottom) coordinates of a bounding box by a segmentation map

    :param segm: Segmentation map (0 - corresponds to the background)
    :return: (left, top, right, bottom) coordinates of the bounding box around human silhouette
    """
    H_nnz = np.nonzero(segm.sum(axis=1))[0]
    W_nnz = np.nonzero(segm.sum(axis=0))[0]

    t = H_nnz[0]
    l = W_nnz[0]

    b = H_nnz[-1]
    r = W_nnz[-1]

    return l, t, r, b


def get_ltrb_from_segm_and_adjust(segm, scale_bbox_value, h, w):
    """
    Calculate (left, top, right, bottom) coordinates of a bounding box by a segmentation map
    and rescale it with scale_bbox_value

    :param segm: Segmentation map (0 - corresponds to the background)
    :param scale_bbox_value: Scale factor of bounding box
    :param h: Height of the image
    :param w: Width of the image
    :return: Bounding box
    """
    ltrb = get_ltrb_from_segm(segm)
    ltrb = scale_bbox(ltrb, scale_bbox_value)
    ltrb = correct_bbox(ltrb, h, w)
    return ltrb


def get_rotation_matrix(angle, axis='x'):
    """
    Make rotation matrix to rotate vertices around selected axis

    :param angle: Angle in radians
    :param axis: Axis around which to rotate
    :return: Rotation matrix
    """
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unkown axis {axis}")


def rotate_verts(verts, angle, K, axis='y'):
    """
    Rotate vertices around selected axis

    :param verts: Vertices to rotate
    :param angle: Angle in radians
    :param K: Camera parameters matrix
    :param axis: Axis around which to rotate
    :return: Rotated copy of vertices
    """
    rotation_matrix = get_rotation_matrix(angle, axis=axis)
    rotation_matrix = torch.from_numpy(rotation_matrix).type(torch.float32).to(verts.device).unsqueeze(0)

    verts_rot = verts.clone()
    K_inv = torch.inverse(K)
    verts_rot = verts_rot.bmm(K_inv.transpose(1, 2))

    mean_point = torch.mean(verts_rot, dim=1)
    verts_rot -= mean_point

    verts_rot = verts_rot.bmm(rotation_matrix.transpose(1, 2))

    verts_rot += mean_point
    verts_rot = verts_rot.bmm(K.transpose(1, 2))

    return verts_rot


def crop_verts(verts, K, ltrb):
    """
    Modify vertices coordinates and calibration matrix according to bounding box

    :param verts: Tensor of vertices
    :param K: Calibration matrix
    :param ltrb: Bounding box (left, top, right, bottom)
    :return: Corrected vertices and calibration matrix
    """
    l, t, r, b = ltrb
    # it's supposed that it smplifyx's verts are in trivial camera coordinates
    fx, fy, cx, cy = 1.0, 1.0, 0.0, 0.0
    # crop
    cx, cy = cx - l, cy - t

    # update verts
    K_upd = np.eye(3)

    K_upd[0, 0] = fx
    K_upd[1, 1] = fy
    K_upd[0, 2] = cx
    K_upd[1, 2] = cy

    verts_cropped = verts @ K_upd.T
    K_cropped = K_upd @ K

    return verts_cropped, K_cropped


def correct_bbox(ltrb, h, w):
    """
    Correct bounding box coordinates according to image width and height

    :param ltrb: Bounding box set as (left, top, right,  bottom)
    :param h: Height of the image
    :param w: Width of the image
    :return: Corrected bounding box coordinates
    """
    l, t, r, b = ltrb

    l = max(0, l)
    t = max(0, t)
    r = min(w, r)
    b = min(h, b)

    return l, t, r, b


def combine_ltrb(ltrb1, ltrb2):
    """
    Combine two bounding boxes (e.g from segmentation and vertices)

    :param ltrb1: (left, top, right, bottom) coordinates of the first bounding box
    :param ltrb2: (left, top, right, bottom) coordinates of the second bounding box
    :return: Merged bounding box
    """

    l1, t1, r1, b1 = ltrb1
    l2, t2, r2, b2 = ltrb2

    l = min(l1, l2)
    t = min(t1, t2)
    r = max(r1, r2)
    b = max(b1, b2)
    return l, t, r, b


def upd_cropmask(crop_mask, verts):
    """
    Update crop mask according to vertices coordinates

    :param crop_mask: Crop mask is used for removing invisible in ground truth parts from the rendering result
    :param verts: Model vertices in homogeneous coordinates
    :return: Adjusted crop mask
    """
    verts_proj = verts.copy()
    verts_proj[:, :2] /= verts_proj[:, 2:]
    vx = verts_proj[:, 0]
    vy = verts_proj[:, 1]

    crop_mask = (crop_mask > 0).astype(np.float32)

    h, w = crop_mask.shape
    l, t, r, b = get_ltrb_from_segm(crop_mask)

    add_l = (vx < l).sum() == 0
    add_t = (vy < t).sum() == 0
    add_r = (vx > r).sum() == 0
    add_b = (vy > b).sum() == 0

    if add_l:
        fr = 0 if add_t else t
        to = h if add_b else b + 1
        crop_mask[fr:to, 0:l] = 1
    if add_r:
        fr = 0 if add_t else t
        to = h if add_b else b + 1
        crop_mask[fr:to, r:] = 1
    if add_t:
        crop_mask[:t, l:r + 1] = 1
    if add_b:
        crop_mask[b:, l:r + 1] = 1

    return crop_mask


def load_colored_uv_map(colored_uv_map, binary=False):
    """
    Load additional information about UV map from image file

    :param colored_uv_map: Path to the image
    :return: Tensor with loaded image. None if name not provided
    """
    if colored_uv_map:
        colored_uv_map = cv2.imread(colored_uv_map)[..., ::-1]
        colored_uv_map = cv2.resize(colored_uv_map, (256, 256))
        if binary:
            colored_uv_map = colored_uv_map[..., 0:1] / 255
        colored_uv_map = torch.FloatTensor(colored_uv_map.copy()).permute(2, 0, 1)
        return colored_uv_map
    return None
