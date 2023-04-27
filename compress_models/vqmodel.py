from torch import nn

from compress_models.diffusion_model import Encoder, Decoder
from compress_models.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=False,
        )
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z, return_loss=False):
        quant, emb_loss, info = self.quantize(z)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        if return_loss:
            return dec, quant, emb_loss, info
        else:
            return dec

    def forward(self, data_dict, compress_only=False):
        source_texture = data_dict['ntexture'].detach()
        z = self.encode(data_dict['ntexture'])

        if compress_only:
            return {'compressed': z}

        dec, quant, diff, _ = self.decode(z, return_loss=True)
        result = {
            'src_ntexture': source_texture,
            'ntexture': dec,
            'codebook_loss': diff,
            'compressed': z,
        }
        return result
