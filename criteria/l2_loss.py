import torch

from utils.image_utils import get_lossmask


class L2Loss(torch.nn.Module):
    def __init__(self, weight=1., intersection=False):
        super().__init__()
        self.weight = weight
        self.intersection = intersection
        self.loss_fn_l2 = torch.nn.MSELoss()

    def forward(self, data_dict):
        synth_images = data_dict['fake_rgb']
        target_images = data_dict['real_rgb']
        lossmask = get_lossmask(data_dict['uv'], data_dict['real_segm'], intersection=self.intersection)

        loss = self.loss_fn_l2(
            target_images * lossmask,
            synth_images * lossmask
        ) * self.weight
        return loss
