import torch
import torch.nn as nn
from .ncsnpp_utils import layers, layerspp, normalization
default_initializer = layers.default_init
default_init = layers.default_init

class dat_trans_merge_crm(nn.Module):
    def __init__(self,
                 nf = 128,
                 fourier_scale = 16,
                 conditional = True,
                 ):
        super().__init__()
        self.en_ri = dense_encoder()
        self.en_mag = dense_encoder_mag()
        self.aia_trans_merge = AIA_Transformer_merge(128, 64, num_layers=4)
        self.aham = AHAM_ori(input_channel=64)
        self.aham_mag = AHAM_ori(input_channel=64)

        self.de1 = dense_decoder()
        self.de2 = dense_decoder()
        self.de_mag_mask = dense_decoder_masking()

        
        time_emb_layers = []
        time_emb_layers.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
        embed_dim = 2 * nf

        self.conditional = conditional
        if conditional:
            time_emb_layers.append(nn.Linear(embed_dim, nf * 4))
            time_emb_layers[-1].weight.data = default_initializer()(time_emb_layers[-1].weight.shape)
            nn.init.zeros_(time_emb_layers[-1].bias)
            time_emb_layers.append(nn.Linear(nf * 4, nf * 4))
            time_emb_layers[-1].weight.data = default_initializer()(time_emb_layers[-1].weight.shape)
            nn.init.zeros_(time_emb_layers[-1].bias)
        self.time_emb = nn.Sequential(*time_emb_layers)
        


    def forward(self, x, time_cond):
        batch_size, _, seq_len, _ = x.shape
        x, y = x[:,0], x[:,1]
        x = torch.cat([x.real, x.imag], dim=1)  # BCTF
        y = torch.cat([y.real, y.imag], dim=1)  # BCTF
        x_r_input, x_i_input = x[:,0,:,:], x[:,1,:,:]
        y_r_input, y_i_input = y[:,0,:,:], y[:,1,:,:]
        x_mag_ori, x_phase_ori = torch.norm(x, dim=1), torch.atan2(x[:, -1, :, :], x[:, 0, :, :])
        y_mag_ori, y_phase_ori = torch.norm(y, dim=1), torch.atan2(y[:, -1, :, :], y[:, 0, :, :])
        x_mag = x_mag_ori.unsqueeze(dim = 1)
        y_mag = y_mag_ori.unsqueeze(dim = 1)
        input_ri = torch.cat([x_r_input, x_i_input, y_r_input, y_i_input], dim=1)
        input_mag = torch.cat([x_mag, y_mag], dim=1)

        used_sigmas = time_cond
        temb = self.time_emb[0](torch.log(used_sigmas))
        if self.conditional:
            temb = self.time_emb[1](temb)
            temb = self.time_emb[2](self.act(temb))

        


        # ri/mag components enconde+ aia_transformer_merge
        x_ri = self.en_ri(x) #BCTF
        x_mag_en = self.en_mag(x_mag)
        x_last_mag, x_outputlist_mag, x_last_ri, x_outputlist_ri  = self.aia_trans_merge(x_mag_en, x_ri)  # BCTF, #BCTFG

        x_ri = self.aham(x_outputlist_ri) #BCT
        x_mag_en = self.aham_mag(x_outputlist_mag)  # BCTF
        x_mag_mask = self.de_mag_mask(x_mag_en)
        x_mag_mask = x_mag_mask.squeeze(dim=1)

        # real and imag decode
        x_real = self.de1(x_ri)
        x_imag = self.de2(x_ri)
        x_real = x_real.squeeze(dim = 1)
        x_imag = x_imag.squeeze(dim=1)
        # magnitude and ri components interaction

        x_mag_out=x_mag_mask * x_mag_ori
        x_r_out,x_i_out = (x_mag_out * torch.cos(x_phase_ori) + x_real), (x_mag_out * torch.sin(x_phase_ori)+ x_imag)

        x_com_out = torch.stack((x_r_out,x_i_out),dim=1)

        return x_com_out


class dense_encoder(nn.Module):
    def __init__(self, act, width =64, temb_dim=None):
        super(dense_encoder, self).__init__()
        self.in_channels = 2
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, width)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
            self.act = act
        self.enc_dense1 = DenseBlock(act, 161, 4, self.width, temb_dim=temb_dim) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x, temb=None):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        if temb is not None:
            out += self.Dense_0(self.act(temb))[:, :, None, None]
        out = self.enc_dense1(out, temb)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x

class dense_encoder_mag(nn.Module):
    def __init__(self, act, width =64, temb_dim=None):
        super(dense_encoder_mag, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.width = width
        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 512]
        self.inp_norm = nn.LayerNorm(161)
        self.inp_prelu = nn.PReLU(self.width)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, width)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
            self.act = act
        self.enc_dense1 = DenseBlock(act, 161, 4, self.width, temb_dim=temb_dim) # [b, 64, nframes, 512]
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 2))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(80)
        self.enc_prelu1 = nn.PReLU(self.width)

    def forward(self, x, temb=None):
        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, T, F]
        if temb is not None:
            out += self.Dense_0(self.act(temb))[:, :, None, None]
        out = self.enc_dense1(out)   # [b, 64, T, F]
        x = self.enc_prelu1(self.enc_norm1(self.enc_conv1(out)))  # [b, 64, T, F]
        return x


class dense_decoder(nn.Module):
    def __init__(self, width =64):
        super(dense_decoder, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.pad = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 0, 0, 0), value=0.)
        self.width =width
        self.dec_dense1 = DenseBlock(80, 4, self.width)
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=2)
        self.dec_norm1 = nn.LayerNorm(161)
        self.dec_prelu1 = nn.PReLU(self.width)

        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.dec_dense1(x)
        out = self.dec_prelu1(self.dec_norm1(self.pad1(self.dec_conv1(self.pad(out)))))

        out = self.out_conv(out)
        out.squeeze(dim=1)
        return out


class DenseBlock(nn.Module): #dilated dense block
    def __init__(self, act, input_size, depth=5, in_channels=64, temb_dim=None):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))
            if temb_dim is not None:
                self.act = act
                Dense_0 = nn.Linear(temb_dim, self.in_channels)
                Dense_0.weight.data = default_init()(Dense_0.weight.shape)
                nn.init.zeros_(Dense_0.bias)
                setattr(self, 'temb{}'.format(i + 1), Dense_0)
    def forward(self, x, temb=None):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            if temb is not None:
                out = out + getattr(self, 'temb{}'.format(i + 1))(self.act(temb))[:, :, None, None]
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class SPConvTranspose2d(nn.Module): #sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out