"""
Implementation of rasterizer based on minimal_pytorch_rasterizer
"""

import minimal_pytorch_rasterizer
import numpy as np
import torch


class Rasterizer(torch.nn.Module):
    """
    Rasterizer to project vertices with pinhole camera
    """

    def __init__(self, H, W, faces_path, vertices_values_path=None):
        """
        Initialize camera faces connections and camera parameters for the rasterizer

        :param H: Height of resulting rasterization
        :param W: Width of resulting rasterization
        :param faces_path: Path to numpy file with vertices indexes to make faces
        :param vertices_values_path: Path to numpy file with UV coordinates for each vertex
        """
        super().__init__()
        faces_cpu = np.load(faces_path)

        self.faces = torch.nn.Parameter(
            torch.tensor(faces_cpu, dtype=torch.int32).contiguous(),
            requires_grad=False,
        )

        if vertices_values_path is not None:
            uv_cpu = np.load(vertices_values_path)
            self.vertice_values = torch.nn.Parameter(
                torch.tensor(uv_cpu, dtype=torch.float32).contiguous(),
                requires_grad=False,
            )
        else:
            self.vertice_values = None

        self.pinhole = minimal_pytorch_rasterizer.Pinhole2D(
            fx=1, fy=1,
            cx=0, cy=0,
            h=H, w=W
        )
        self.H = H
        self.W = W

    def set_vertice_values(self, vertices_values):
        """
        Set "color" values for vertices to visualize in rasterizer.

        :param vertices_values: Values for each vertice
        :return:
        """
        self.vertice_values = torch.nn.Parameter(
            torch.tensor(vertices_values, dtype=torch.float32).to(self.faces.device),
            requires_grad=False,
        )

    def forward(self, data_dict, norm=True, negbg=True, return_mask=True):
        """
        Calculate UV-render for vertices coordinates

        :param data_dict: Data_dict with vertices coordinat
        :param norm: Flag to normalize output UV-render values in [-1, 1] range
        :param negbg: Set background as a negative value
        :param return_mask: Flag to return binarized version of UV-render
        :return: Data_dict with UV-render (and optionally a mask)
        """
        verts = data_dict['verts']
        N = verts.shape[0]

        uvs = []
        for i in range(N):
            v = verts[i]
            uv = minimal_pytorch_rasterizer.project_mesh(v, self.faces, self.vertice_values, self.pinhole)
            uvs.append(uv)

        uvs = torch.stack(uvs, dim=0).permute(0, 3, 1, 2)
        mask = (uvs > 0).sum(dim=1, keepdim=True).float().clamp(0., 1.)

        if norm:
            uvs = (uvs * 2 - 1.)

        if negbg:
            uvs = uvs * mask - 10 * torch.logical_not(mask)

        out_dict = {}
        out_dict['uv'] = uvs
        if return_mask:
            out_dict['uv_mask'] = mask

        return out_dict
