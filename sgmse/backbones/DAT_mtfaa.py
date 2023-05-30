import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList
from .ncsnpp_utils import layers, layerspp, normalization
default_initializer = layers.default_init
default_init = layers.default_init
get_act = layers.get_act
from .shared import BackboneRegistry


from typing import List

from .mtfaa_utils.tfcm import TFCM
from .mtfaa_utils.asa import ASA
from .mtfaa_utils.phase_encoder import PhaseEncoder
from .mtfaa_utils.f_sampling import FD, FU


def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]


eps = 1e-10

@BackboneRegistry.register("DAT_MTFAA")
class DAT_MTFAA(nn.Module):
    def __init__(self,
                 Co="48,96,192,384",
                 O="1,1,1,1",
                 causal=False,
                 bottleneck_layer=2,
                 tfcm_layer=8,
                 nf = 128,
                 fourier_scale = 16,
                 conditional = True,
                 nonlinearity = 'swish',
                 scale_by_sigma = True,
                 **unused_kwargs
                 ):
        super(DAT_MTFAA, self).__init__()
        time_emb_layers = []
        time_emb_layers.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
        embed_dim = 2 * nf
        self.act = act = get_act(nonlinearity)
        self.conditional = conditional
        if conditional:
            time_emb_layers.append(nn.Linear(embed_dim, nf * 4))
            time_emb_layers[-1].weight.data = default_initializer()(time_emb_layers[-1].weight.shape)
            nn.init.zeros_(time_emb_layers[-1].bias)
            time_emb_layers.append(nn.Linear(nf * 4, nf * 4))
            time_emb_layers[-1].weight.data = default_initializer()(time_emb_layers[-1].weight.shape)
            nn.init.zeros_(time_emb_layers[-1].bias)
        self.time_emb = nn.Sequential(*time_emb_layers)

        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [4] + parse_1dstr(Co)
        C_de = [4] + parse_1dstr(Co)
        O = parse_1dstr(O)
        for idx in range(len(C_en)-1):
            self.encoder_fd.append(
                FD(act, nf * 4, C_en[idx], C_en[idx+1]),
            )
            self.encoder_bn.append(
                nn.ModuleList([
                    TFCM(act, nf * 4, C_en[idx+1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[idx+1], causal=causal),
                ])
            )

        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.ModuleList([
                    TFCM(act, nf * 4, C_en[-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_en[-1], causal=causal),
                ])
            )

        for idx in range(len(C_de)-1, 0, -1):
            self.decoder_fu.append(
                FU(act, nf * 4, C_de[idx], C_de[idx-1], O=(O[idx-1], 0)),
            )
            self.decoder_bn.append(
                nn.ModuleList([
                    TFCM(act, nf * 4, C_de[idx-1], (3, 3),
                         tfcm_layer=tfcm_layer, causal=causal),
                    ASA(C_de[idx-1], causal=causal),
                ])
            )
        # MEA is causal, so mag_t_dim = 1.
        self.output_layer = nn.Conv2d(4, 2, kernel_size=(3, 3), padding=(1, 1))

        self.scale_by_sigma = scale_by_sigma

            
        
        # self.real_mask = nn.Conv2d(4, 2, kernel_size=(3, 1), padding=(1, 0))
        # self.imag_mask = nn.Conv2d(4, 2, kernel_size=(3, 1), padding=(1, 0))
        # kernel = torch.eye(mag_f_dim)
        # kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        # self.register_buffer('kernel', kernel)
        # self.mag_f_dim = mag_f_dim
        # self.mag_t_dim = mag_t_dim

    @staticmethod
    def add_argparse_args(parser):
        # TODO: add additional arguments of constructor, if you wish to modify them.
        return parser

        
    def forward(self, x, time_cond):
        x = torch.stack((x[:,0,:,:].real, x[:,0,:,:].imag,
            x[:,1,:,:].real, x[:,1,:,:].imag), dim=1) # B4FT
        out = x
        
        used_sigmas = time_cond
        temb = self.time_emb[0](torch.log(used_sigmas))
        if self.conditional:
            temb = self.time_emb[1](temb)
            temb = self.time_emb[2](self.act(temb))
        
        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out, temb)
            encoder_out.append(out)
            out = self.encoder_bn[idx][0](out, temb)
            out = self.encoder_bn[idx][1](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx][0](out, temb)
            out = self.bottleneck[idx][1](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1-idx], temb)
            out = self.decoder_bn[idx][0](out, temb)
            out = self.decoder_bn[idx][1](out)

        
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            out = out / used_sigmas

        out = self.output_layer(out)
        out = torch.permute(out, (0, 2, 3, 1)).contiguous()
        out = torch.view_as_complex(out)[:,None, :, :]


        # stage 1
        # mag_mask = self.mag_mask(out)
        # mag_pad = F.pad(
        #     mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        # mag = F.conv2d(mag_pad, self.kernel)
        # mag = mag * mag_mask.sigmoid()
        # mag = mag.sum(dim=1)
        # # stage 2
        # real_mask = self.real_mask(out).squeeze(1)
        # imag_mask = self.imag_mask(out).squeeze(1)

        # mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, eps))
        # pha_mask = torch.atan2(imag_mask+eps, real_mask+eps)
        # real = mag * mag_mask.tanh() * torch.cos(pha+pha_mask)
        # imag = mag * mag_mask.tanh() * torch.sin(pha+pha_mask)
        return out


def test_nnet():
    # noise supression (microphone, )
    nnet = DAT_MTFAA()
    x = torch.view_as_complex(torch.randn(2, 257, 200, 2))
    y = torch.view_as_complex(torch.randn(2, 257, 200, 2))
    inp = torch.stack((x, y), dim=1)
    time = torch.rand(2)
    output = nnet(inp, time)
    print(output.shape)


if __name__ == "__main__":
    # test_nnet()
    test_nnet()
