import random

import torch
from lpips import lpips

from utils.image_utils import get_lossmask


class BasePerceptualLoss(torch.nn.Module):
    def __init__(self, loss_fn, weight, intersection):
        super().__init__()
        self.weight = weight
        self.loss_fn = loss_fn
        self.intersection = intersection

    def forward(self, data_dict):
        self.loss_fn.to(data_dict['fake_rgb'].device)
        synth_images = data_dict['fake_rgb']
        target_images = data_dict['real_rgb']
        lossmask = get_lossmask(data_dict['uv'], data_dict['real_segm'], intersection=self.intersection)

        verts = data_dict['verts'][0]
        verts_projected = (verts / (verts[:, 2:]))[:, :2]
        point_on_model = random.choice(verts_projected)
        cx, cy = point_on_model
        j = max(int(cx), 128)
        i = max(int(cy), 128)

        i = min(i, synth_images.shape[2] - 129)
        j = min(j, synth_images.shape[3] - 129)

        sub_synth = synth_images[:, :, i - 128: i + 128, j - 128: j + 128]
        sub_target = target_images[:, :, i - 128: i + 128, j - 128: j + 128]
        sum_mask = lossmask[:, :, i - 128: i + 128, j - 128: j + 128]

        loss = self.loss_fn(
            sub_target * sum_mask,
            sub_synth * sum_mask
        ).mean() * self.weight
        return loss


class PerceptualLoss(BasePerceptualLoss):
    def __init__(self, weight=1., intersection=False):
        loss_fn = lpips.LPIPS(net='vgg')
        super().__init__(loss_fn, weight, intersection)
