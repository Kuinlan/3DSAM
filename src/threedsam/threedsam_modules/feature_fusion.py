import torch
import torch.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class Up(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=True):
        super(Up, self).__init__()

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
        super(FeatureFusion, self).__init__()
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

    def forward(self, feat_8, feat_16, feat_32):
        # update from 32&16 to 8
        feat_32_8 = self.up_32_8(feat_32)
        feat_16_8 = self.up_16_8(feat_16)
        feat_out8 = feat_8 + feat_32_8 + feat_16_8

        # update from 8 to 16&32
        feat_8_16 = self.down_8_16(feat_8)
        feat_8_32 = self.down_8_32(feat_8)
        feat_out16 = feat_16 + feat_8_16
        feat_out32 = feat_32 + feat_8_32

        return feat_out8, feat_out16, feat_out32

