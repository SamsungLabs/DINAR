import torch

from utils.networks_utils.sg2_modules import d_r1_loss


class R1Regularization(torch.nn.Module):
    def __init__(self, weight=1., reg_every=16):
        super().__init__()
        self.weight = weight
        self.reg_every = reg_every

        self.count = reg_every

    def is_called(self):
        self.count -= 1
        if self.count == 0:
            self.count = self.reg_every
            return True
        else:
            return False

    def forward(self, data_dict):
        self.count = self.reg_every
        real_img = data_dict['disc_input']
        real_score = data_dict['real']['score']

        r1_loss = d_r1_loss(real_score, real_img)
        r1_loss = (self.weight / 2) * self.reg_every * r1_loss + 0 * real_score[0]
        return r1_loss[0]
