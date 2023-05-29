"""
TCN modules (TCM) -> TFCN modules (TFCM).

shmzhang@aslp-npu.org, 2022
"""

import torch as th
import torch.nn as nn

from ..ncsnpp_utils import layers, layerspp, normalization
default_initializer = layers.default_init
default_init = layers.default_init

class TFCM_Block(nn.Module):
    def __init__(self,
                 act,
                 temb_dim,
                 cin=24,
                 K=(3, 3),
                 dila=1,
                 causal=True,
                 ):
        super(TFCM_Block, self).__init__()
        self.Dense_0 = nn.Linear(temb_dim, cin)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = act

        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=(1, 1)),
            # nn.BatchNorm2d(cin),
            nn.GroupNorm(num_groups=1,num_channels=cin),
            nn.PReLU(cin),
        )
        dila_pad = dila * (K[1] - 1)
        if causal:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                # nn.BatchNorm2d(cin),
                nn.GroupNorm(num_groups=1,num_channels=cin),
                nn.PReLU(cin)
            )
        else:
            # update 22/06/21, add groups for non-casual
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad//2, dila_pad//2, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                # nn.BatchNorm2d(cin),
                nn.GroupNorm(num_groups=1,num_channels=cin),
                nn.PReLU(cin)
            )
        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
        self.causal = causal
        self.dila_pad = dila_pad

    def forward(self, inps, temb):
        """
            inp: B x C x F x T
        """
        outs = self.pconv1(inps)
        outs = outs + self.Dense_0(self.act(temb))[:, :, None, None]
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs + inps


class TFCM(nn.Module):
    def __init__(self,
                 act,
                 temb_dim,
                 cin=24,
                 K=(3, 3),
                 tfcm_layer=6,
                 causal=True,
                 ):
        super(TFCM, self).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(
                TFCM_Block(act, temb_dim, cin, K, 2**idx, causal=causal)
            )

    def forward(self, inp, temb):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out, temb)
        return out


def test_tfcm():
    nnet = TFCM(24)
    inp = th.randn(2, 24, 256, 101)
    out = nnet(inp)
    print(out.shape)


if __name__ == "__main__":
    test_tfcm()
