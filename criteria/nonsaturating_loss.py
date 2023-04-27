import torch

from utils.networks_utils.sg2_modules import g_nonsaturating_loss, d_logistic_loss


class Adversarial(torch.nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight

    def forward(self, data_dict):
        if 'real' in data_dict:
            fake_score = data_dict['fake']['score']
            real_score = data_dict['real']['score']
            d_loss = d_logistic_loss(real_score, fake_score) * self.weight
            loss_dict_D = d_loss.mean()
            return loss_dict_D
        else:
            fake_score = data_dict['fake']['score']
            g_loss = g_nonsaturating_loss(fake_score) * self.weight
            loss_dict_G = g_loss.mean()
            return loss_dict_G
