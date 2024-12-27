import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from einops.einops import rearrange


class BasicBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 ):
        super(BasicBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out

class DeformConv(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1):
        super(DeformConv, self).__init__()

        self.conv_offset = nn.Conv2d(in_channels, 2 * groups * kernel_size * kernel_size, 
                                kernel_size=kernel_size, stride=stride, padding=padding)
        offset_init = torch.zeros(2 * groups * kernel_size * kernel_size, in_channels, kernel_size, kernel_size)
        self.conv_offset.weight = torch.nn.Parameter(offset_init)

        self.conv_mask = nn.Conv2d(in_channels, groups * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        mask_init = torch.zeros(groups * kernel_size * kernel_size, in_channels, kernel_size, kernel_size) + 0.5
        self.conv_mask.weight = torch.nn.Parameter(mask_init)

        self.deform_conv = DeformConv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation,
                                        groups=groups)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = self.deform_conv(x, offset, mask=mask)

        return out
        

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            x: (B, C, H, W)
        """
        x1 = self.aspp1(x)      # (B, C, H, W)
        x2 = self.aspp2(x)      # (B, C, H, W)
        x3 = self.aspp3(x)      # (B, C, H, W)
        x4 = self.aspp4(x)      # (B, C, H, W)
        x5 = self.global_avg_pool(x)    # (B, C, 1, 1)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)      # (B, C, H, W)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)      # (B, 5*C, H, W)

        x = self.conv1(x)   # (B, C, H, W)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# class SELikeModule(nn.Module):
#     def __init__(self, in_channel=256, feat_channel=256, intrinsic_channel=6):
#         super(SELikeModule, self).__init__()
#         self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
#         # self.fc = nn.Sequential(
#         #     nn.BatchNorm1d(intrinsic_channel),
#         #     nn.Linear(intrinsic_channel, feat_channel),
#         #     nn.Sigmoid())

#     def forward(self, x, depth_embed):
#         """
#         Args:
#             x: (B, C, H, W)
#             depth_embed: (B, C, H, W)

#         Returns:
#             x:  (B*N_view, C, H, W)
#         """
#         x = self.input_conv(x)  # (B, C, H, W)
#         y = depth_embed
#         return x + y

class DepthEmbedModule(nn.Module):
    def __init__(self, in_channel=512, out_channel=256):
        super(DepthEmbedModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=False)
        )

        self.norm = nn.LayerNorm(out_channel)

    def forward(self, x, depth_embed):
        _, _, H, W = x.shape
        x = rearrange(x, 'n c h w -> n (h w) c').contiguous()
        depth_embed = rearrange(depth_embed, 'n c h w -> n (h w) c').contiguous()
        out = self.norm(self.mlp(torch.cat([x, depth_embed], dim=2)))
        out = rearrange(out, 'n (h w) c -> n c h w', h=H, w=W).contiguous()
        
        return out

class SELikeModule(nn.Module):
    def __init__(self, in_channel=256, feat_channel=256, intrinsic_channel=6):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid())

    def forward(self, x, cam_params):
        """
        Args:
            x: (B, C_in, H, W)
            cam_params: (B, 6)

        Returns:
            x:  (B, C, H, W)
        """
        x = self.input_conv(x)  # (B, C, H, W)
        b, c, _, _ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)    # (B, C, 1, 1)
        return x * y.expand_as(x)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm = nn.BatchNorm2d(out_channels)
        
        # 激活函数（ReLU）
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.norm(self.conv(x))
        x = self.relu(x)

        return x


class CameraAwareDepthNet(nn.Module):
    def __init__(self, config, num_params=6):
        super(CameraAwareDepthNet, self).__init__()
        self.in_channels = config['in_channels']
        self.context_channels = config['context_channels']
        self.depth_channels = config['depth_channels']
        self.mid_channels = config['mid_channels']

        self.rel_depth_embed = nn.Embedding(256, 256)

        self.depth_max = config['depth_max']
        self.depth_min = config['depth_min']
        self.depth_num = config['num_depth_bins']

        self.with_context_encoder = config['with_context_encoder']
        self.with_depth_correction = config['with_depth_correction']

        self.merge = DepthEmbedModule(in_channel=512, out_channel=256)
        if self.mid_channels is not None:
            mid_channels = self.mid_channels
            self.reduce_conv = ConvModule(
                        in_channels=self.in_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        )
        else:
            mid_channels = self.in_channels
            self.reduce_conv = None

        context_channels = self.context_channels
        if self.with_context_encoder:
            self.context_conv = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        
        # 对相机参数进行嵌入
        self.se = SELikeModule(in_channel=self.mid_channels, feat_channel=self.mid_channels,
                               intrinsic_channel=num_params)

        # 深度修正: DCN
        depth_channels = self.depth_channels
        if self.with_depth_correction:
            self.depth_stem = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
                ASPP(mid_channels, mid_channels),
            )
            self.depth_prob_conv = nn.Sequential(
                DeformConv(in_channels=mid_channels,
                             out_channels=mid_channels,
                             kernel_size=3,
                             padding=1,
                             groups=4),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.depth_stem = torch.nn.Identity()
            self.depth_prob_conv = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)

        self.with_pgd = config['with_pgd']
        # 启用加权的深度直接回归估计
        if self.with_pgd:
            self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))
            self.depth_direct_conv = nn.Sequential(
                DeformConv(in_channels=mid_channels,
                             out_channels=mid_channels,
                             kernel_size=3,
                             padding=1,
                             groups=4),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
            )

        index = torch.arange(start=0, end=self.depth_channels, step=1).float()
        bin_size = (self.depth_max - self.depth_min) / (self.depth_num - 1)
        depth_bin = self.depth_min + bin_size * index
        self.register_buffer('project', depth_bin)

    def forward(self, feat0, feat1, data):
        """
        Args:
            feat0: img feature map  (B, C, H, W)
            feat1: img feature map  (B, C, H, W)
            data: Dict
        Returns:
            depth_prob0:  (B, D, H, W)
            depth_prob1:  (B, D, H, W)
            depth_direct0: (B, H, W)
            depth_direct1: (B, H, W)
            context0: (B, C_context, H, W)
            context1: (B, C_context, H, W)
        """
        B, _, H, W = feat0.shape
        rel_depth0 = data['rel_depth0']  # (B, H, W)
        rel_depth1 = data['rel_depth1']
        intrinsic0 = data['K0'][:, :2, :].clone().contiguous()  
        intrinsic1 = data['K1'][:, :2, :].clone().contiguous()  
        intrinsic0 = intrinsic0.view(B, -1)  # (B, 6)
        intrinsic1 = intrinsic1.view(B, -1)

        depth_embed0 = self.interpolate_depth_embed(rel_depth0)  # (N, H, W, C)
        depth_embed1 = self.interpolate_depth_embed(rel_depth1) 

        feat0 = self.merge(feat0, depth_embed0)
        feat1 = self.merge(feat1, depth_embed1)

        # (B*N_view, C, H, W) --> (B*N_view, C_mid, H, W)
        # x 用于估计深度
        # context 用于产生深度信息
        if self.reduce_conv is not None:
            feat0 = self.reduce_conv(feat0)
            feat1 = self.reduce_conv(feat1)

        context0 = self.context_conv(feat0)  # (B*N_view, C_context, H, W)
        context1 = self.context_conv(feat1) 
        depth0 = self.se(feat0, intrinsic0)  # (B, C_mid, H, W)
        depth1 = self.se(feat1, intrinsic1)  # (B, C_mid, H, W)

        if not self.with_pgd:
            depth_stem0 = self.depth_stem(depth0)
            depth_stem1 = self.depth_stem(depth1)
            depth_prob0 = self.depth_prob_conv(depth_stem0)  # (B, D, H, W)
            depth_prob1 = self.depth_prob_conv(depth_stem1) 
            return depth_prob0, depth_prob1, context0, context1
        else:
            depth_stem0 = self.depth_stem(depth0)
            depth_stem1 = self.depth_stem(depth1)
            depth_prob0 = self.depth_prob_conv(depth_stem0)
            depth_prob1 = self.depth_prob_conv(depth_stem1)
            depth_direct0 = self.depth_direct_conv(depth_stem0)
            depth_direct1 = self.depth_direct_conv(depth_stem1)

        self.depth_score0 = depth_prob0   # 未经过softmax
        depth_prob0 = depth_prob0.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B, H, W, D) --> (B*H*W, D)
        depth_prob_val0 = self.integral(depth_prob0)      # (B*H*W, )
        depth_map_pred0 = depth_prob_val0

        self.depth_score1 = depth_prob1 
        depth_prob1 = depth_prob1.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B, H, W, D) --> (B*H*W, D)
        depth_prob_val1 = self.integral(depth_prob1)      # (B*H*W, )
        depth_map_pred1 = depth_prob_val1

        if self.with_pgd:
            sig_alpha = torch.sigmoid(self.fuse_lambda)

            depth_direct_val0 = depth_direct0.view(-1)      # (B*H*W, )
            depth_pgd_fuse0 = sig_alpha * depth_direct_val0 + (1 - sig_alpha) * depth_prob_val0
            depth_map_pred0 = depth_pgd_fuse0

            depth_direct_val1 = depth_direct1.view(-1)      # (B*H*W, )
            depth_pgd_fuse1 = sig_alpha * depth_direct_val1 + (1 - sig_alpha) * depth_prob_val1
            depth_map_pred1 = depth_pgd_fuse1
        else:
            # direct depth
            depth_map_pred0 = depth_prob0.exp().view(-1)     # (B*H*W, )
            depth_map_pred1 = depth_prob1.exp().view(-1)     # (B*H*W, )

        depth_map_pred0 = depth_map_pred0.view(B, H, W)
        depth_map_pred1 = depth_map_pred1.view(B, H, W)

        return depth_map_pred0, depth_map_pred1, context0, context1, self.depth_score0, self.depth_score1


    def interpolate_depth_embed(self, depth):
        pos = self.interpolate_1d(depth, self.rel_depth_embed)
        pos = rearrange(pos, 'n h w c -> n c h w')
        return pos
    
    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings-1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta # [N H W C]

    def integral(self, depth_pred):
        """
        Args:
            depth_pred: (N, D)
        Returns:
            depth_val: (N, )
        """
        depth_score = F.softmax(depth_pred, dim=-1)     # (N, D)
        depth_val = F.linear(depth_score, self.project.type_as(depth_score))  # (N, D) * (D, )  --> (N, )
        return depth_val