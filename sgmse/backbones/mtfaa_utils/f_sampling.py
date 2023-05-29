"""
Frequency Down/Up Sampling.

shmzhang@aslp-npu.org, 2022
"""


import torch as th
import torch.nn as nn
from ..ncsnpp_utils import layers, layerspp, normalization
default_initializer = layers.default_init
default_init = layers.default_init

class FD(nn.Module):
    def __init__(self, act, temb_dim, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0)):
        super(FD, self).__init__()
        self.fd = nn.Sequential(
            nn.Conv2d(cin, cout, K, S, P),
            # nn.BatchNorm2d(cout),
            nn.GroupNorm(num_groups=1,num_channels=cout),
            nn.PReLU(cout)
        )
        self.Dense_0 = nn.Linear(temb_dim, cout)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = act

    def forward(self, x, temb):
        return self.fd(x) + self.Dense_0(self.act(temb))[:, :, None, None]


class FU(nn.Module):
    def __init__(self, act, temb_dim, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0), O=(1, 0)):
        super(FU, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin*2, cin, (1, 1)),
            # nn.BatchNorm2d(cin),
            nn.GroupNorm(num_groups=1,num_channels=cin),
            nn.Tanh(),
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 1)),
            # nn.BatchNorm2d(cout),
            nn.GroupNorm(num_groups=1,num_channels=cout),
            nn.PReLU(cout),
        )
        #  22/06/13 update, add groups = 2
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(cout, cout, K, S, P, O),
            # nn.BatchNorm2d(cout),
            nn.GroupNorm(num_groups=1,num_channels=cout),
            nn.PReLU(cout)
        )
        self.Dense_0 = nn.Linear(temb_dim, cout)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
        nn.init.zeros_(self.Dense_0.bias)
        self.act = act

    def forward(self, fu, fd, temb):
        """
        fu, fd: B C F T
        """
        outs = self.pconv1(th.cat([fu, fd], dim=1))*fd
        outs = self.pconv2(outs)
        outs = outs + + self.Dense_0(self.act(temb))[:, :, None, None]
        outs = self.conv3(outs)
        return outs


def test_fd():
    net = FD(4, 8)
    inps = th.randn(3, 4, 256, 101)
    print(net(inps).shape)


if __name__ == "__main__":
    test_fd()
