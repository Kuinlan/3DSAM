import torch
import torch.nn as nn
from einops.einops import rearrange

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class Up(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                conv1x1(ch_in, ch_out),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(),
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                conv1x1(ch_out, ch_out),
                nn.BatchNorm2d(ch_out),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.conv(self.up(x))
    
        return x
    
class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv1x1(ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.maxpool_conv(x)

        return x

class FeatureFusion(nn.Module):
    """Get features updated from each other. """
    def __init__(self, config):
        super().__init__()
        self.block_dims = config['block_dims']
        self.up_32_8 = nn.Sequential(
            Up(config['block_dims'][4], config['block_dims'][3], bilinear=True),
            Up(config['block_dims'][3], config['block_dims'][2], bilinear=True)
        )
        self.up_16_8 = Up(config['block_dims'][3], config['block_dims'][2], bilinear=True)
        self.down_8_16 = Down(config['block_dims'][2], config['block_dims'][3])
        self.down_8_32 = nn.Sequential(
            Down(config['block_dims'][2], config['block_dims'][3]),
            Down(config['block_dims'][3], config['block_dims'][4])
        )

    def forward(self, feat_8, feat_16, feat_32, data, last_iter=False):
        N, _, _ = feat_8.shape
        hw_c_8 = data['hw0_c_8'] 
        hw_c_16 = data['hw0_c_16'] 
        hw_c_32 = data['hw0_c_32']

        feat_8 = rearrange(feat_8, 'b (h w) c -> b c h w', b=N, h=hw_c_8[0], w=hw_c_8[1], c=self.block_dims[2])
        feat_16 = rearrange(feat_16, 'b (h w) c -> b c h w', b=N, h=hw_c_16[0], w=hw_c_16[1], c=self.block_dims[3])
        feat_32 = rearrange(feat_32, 'b (h w) c -> b c h w', b=N, h=hw_c_32[0], w=hw_c_32[1], c=self.block_dims[4])
        # feat_8 = feat_8.transpose(1, 2).contiguous().view(N, self.block_dims[2], hw_c_8[0], hw_c_8[1])  # (N, C, H, W)
        # feat_16 = feat_16.transpose(1, 2).contiguous().view(N, self.block_dims[3], hw_c_16[0], hw_c_16[1])
        # feat_32 = feat_32.transpose(1, 2).contiguous().view(N, self.block_dims[4], hw_c_32[0], hw_c_32[1])

        # update from 32&16 to 8
        feat_32_8 = self.up_32_8(feat_32)
        feat_16_8 = self.up_16_8(feat_16)
        feat_out8 = feat_8 + feat_32_8 + feat_16_8

        # update from 8 to 16&32
        if not last_iter:
            feat_8_16 = self.down_8_16(feat_8)
            feat_8_32 = self.down_8_32(feat_8)
            feat_out16 = feat_16 + feat_8_16
            feat_out32 = feat_32 + feat_8_32

            feat_out8 = rearrange(feat_8, 'b c h w -> b (h w) c', b=N, h=hw_c_8[0], w=hw_c_8[1], c=self.block_dims[2])
            feat_out16 = rearrange(feat_16, 'b c h w -> b (h w) c', b=N, h=hw_c_16[0], w=hw_c_16[1], c=self.block_dims[3])
            feat_out32 = rearrange(feat_32, 'b c h w -> b (h w) c', b=N, h=hw_c_32[0], w=hw_c_32[1], c=self.block_dims[4])
            # feat_out8 = feat_out8.view(N, self.block_dims[2], -1).transpose(1, 2).contiguous()
            # feat_out16 = feat_out16.view(N, self.block_dims[3], -1).transpose(1, 2).contiguous()
            # feat_out32 = feat_out32.view(N, self.block_dims[4], -1).transpose(1, 2).contiguous()

            return [feat_out8, feat_out16, feat_out32]

        else:
            feat_out8 = feat_out8.view(N, -1, self.block_dims[2])

            return [feat_out8]


