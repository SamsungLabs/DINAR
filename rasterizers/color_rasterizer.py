"""
Customized version of uv_rasterizer.py to rasterize meshes with different "colors" values for each batch element
"""

import minimal_pytorch_rasterizer
import numpy as np
import torch


class ColorsRasterizer(torch.nn.Module):
    """
    Rasterizer to project vertices with pinhole camera
    """
    def __init__(self, H, W, faces_path):
        """
        Initialize camera faces connections and camera parameters for the rasterizer

        :param H: Height of resulting rasterization
        :param W: Width of resulting rasterization
        :param faces_path: Path to numpy file with vertices indexes to make faces
        """
        super().__init__()
        faces_cpu = np.load(faces_path)

        self.faces = torch.nn.Parameter(torch.tensor(faces_cpu, dtype=torch.int32).contiguous(), requires_grad=False)
        self.pinhole = minimal_pytorch_rasterizer.Pinhole2D(
            fx=1, fy=1,
            cx=0, cy=0,
            h=H, w=W
        )

    def forward(self, verts, colors, return_mask=False):
        """
        Rasterize batch of vertices with batch of colors

        :param verts: Batch of meshes set as vertices lists
        :param colors: Batch of colors set as vertices colors list
        :param return_mask: Flag to return binarized version of rasterization
        :return: Rasterized model, [binarized rasterization]
        """
        N = verts.shape[0]

        uvs = []
        for i in range(N):
            v = verts[i]
            c = colors[i]
            uv = minimal_pytorch_rasterizer.project_mesh(v, self.faces, c, self.pinhole)
            uvs.append(uv)

        uvs = torch.stack(uvs, dim=0).permute(0, 3, 1, 2)
        mask = (uvs > 0).sum(dim=1, keepdim=True).float().clamp(0., 1.)
        uvs = uvs * mask

        if return_mask:
            return uvs, mask
        else:
            return uvs
