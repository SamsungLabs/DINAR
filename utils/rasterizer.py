"""
Functions to work with rasterizer and rasterized meshes
"""
import numpy as np
import torch
import cv2


def calc_normals(verts, K, faces):
    """
    Calculate normal vectors to the mesh set as vertices and faces lists

    :param verts: Tensor with vertices
    :param K: Camera calibration tensor
    :param faces: Faces tensor. Each face set as three vertices indices
    :return: Tensor with normal vectors
    """
    K_inv = torch.inverse(K)
    verts_world = verts @ K_inv.transpose(1, 2)

    verts_by_face = []
    for i in range(3):
        vbf = torch.index_select(verts_world, 1, faces[:, i])
        verts_by_face.append(vbf)
    verts_by_face = torch.stack(verts_by_face, dim=2)  # B, N, T, C

    verts_by_face_perm = verts_by_face[:, :, [1, 2, 0]]
    faces_vectors = verts_by_face_perm - verts_by_face  # A->B, B->C, C->A
    faces_vectors_perm = faces_vectors[:, :, [1, 2, 0]]

    cross_products = torch.cross(faces_vectors_perm, faces_vectors, dim=-1)

    verts_normals = torch.zeros_like(verts)
    verts_normals.index_add_(1, faces[..., 0], cross_products[:, :, 0])
    verts_normals.index_add_(1, faces[..., 1], cross_products[:, :, 1])
    verts_normals.index_add_(1, faces[..., 2], cross_products[:, :, 2])

    verts_normals /= torch.norm(verts_normals, dim=-1, keepdim=True)
    return verts_normals


def calculate_cam_Rt(center, direction, right, up):
    """
    Calculate rotation and translation matrices for a camera.
    Camera set as center position and vectors from the center.

    :param center: Camera position in the space
    :param direction: Vector with camera direction
    :param right: Vector directed to the right from the camera
    :param up: Vector directed to the up from the camera
    :return: Stacked rotation and translation matrices
    """

    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    center = center.reshape([-1])
    direction = direction.reshape([-1])
    right = right.reshape([-1])
    up = up.reshape([-1])

    rot_mat = np.eye(3)
    s = right
    s = normalize_vector(s)
    rot_mat[0, :] = s
    u = up
    u = normalize_vector(u)
    rot_mat[1, :] = -u
    rot_mat[2, :] = normalize_vector(direction)
    trans = -np.dot(rot_mat, center)

    return np.hstack((rot_mat, trans[:, None]))


def erode_occlusions(uv, colored_uv_map):
    """
    Apply erosion to the edge of self occluded areas of a model

    :param uv: Rasterized SMPL-X mesh with UV coordinates (UV-render)
    :param colored_uv_map: Texture with color codes for each body parts
    :return: UV-render with deleted areas around occluded regions
    """
    colored_uv_map = colored_uv_map.repeat(uv.shape[0], 1, 1, 1)
    uv_color = torch.nn.functional.grid_sample(colored_uv_map, uv.permute(0, 2, 3, 1))
    uv_color = uv_color.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    erode_masks = []
    for i in range(uv_color.shape[0]):
        edges = cv2.Canny(uv_color[i], 150, 200)
        uv_gray = cv2.cvtColor(uv_color[i], cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(uv_gray, 0, 255, 0)[1]
        contours = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]
        erode_mask = cv2.drawContours(edges.copy(), contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
        erode_mask = cv2.dilate(erode_mask, np.ones((3, 3), dtype=np.uint8), iterations=2)[..., None]
        erode_mask = erode_mask.astype(np.uint8) / 255.
        erode_masks.append(erode_mask)

    erode_mask = torch.FloatTensor(np.stack(erode_masks)).permute(0, 3, 1, 2).to(uv.device)
    res = uv * (1 - erode_mask) + torch.ones_like(uv) * erode_mask * -10.
    return res
