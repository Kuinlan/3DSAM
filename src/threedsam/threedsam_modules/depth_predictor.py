import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoderLayerGeneral
from src.loftr.utils.position_encoding import PositionEncodingSine
from einops.einops import rearrange

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

class CameraAwareDepthPredictor(nn.Module):
    def __init__(self, config, num_params=6):
        super(CameraAwareDepthPredictor, self).__init__()
        self.mid_channels = config['mid_channels']

        self.rel_depth_embed = nn.Embedding(256, 256)

        self.depth_max = config['depth_max']
        self.depth_min = config['depth_min']
        self.depth_num = config['num_depth_bins']

        self.merge = DepthEmbedModule(in_channel=512, out_channel=256)

        d_model = config['d_model']
        feat_m_dim = config['d_model_mid']

        # extrinsic embedding
        self.se = SELikeModule(in_channel=d_model, feat_channel=d_model,
                               intrinsic_channel=num_params)
        # Predict depth info
        d_model = config['d_model']
        feat_m_dim = config['d_model_mid']
        depth_encoder_layer = TransformerEncoderLayerGeneral(d_model, nhead=config['nhead'])
        self.num_layers = config['num_layers']
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(depth_encoder_layer) for _ in range(self.num_layers)]
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(feat_m_dim, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model)
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU()
        )

        depth_channel = config['depth_channels']
        self.depth_prob_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, depth_channel, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=depth_channel),
            nn.ReLU()
        )
        
        self.depth_direct_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, 1, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.pos_encoding = PositionEncodingSine(d_model, temp_bug_fix=True)

        # 启用加权的深度直接回归估计
        self.fuse_lambda = nn.Parameter(torch.tensor(10e-5))
        index = torch.arange(start=0, end=depth_channel, step=1).float()
        bin_size = (self.depth_max - self.depth_min) / (self.depth_num - 1)
        depth_bin = self.depth_min + bin_size * index
        self.register_buffer('project', depth_bin)

    def forward(self, feat_c, feat_m, rel_depth, intrinsic):
        # feature merge
        src_4 = self.downsample(feat_m)
        src_8 = self.proj(feat_c)
        src = (src_4 + src_8) / 2
        
        # introduce rel depth 
        rel_depth_embed = self.interpolate_depth_embed(rel_depth)  # (N, C, H, W)

        # TODO:
        src = self.merge(src, rel_depth_embed) 

        # introduce camera intrinsic
        B, C, H, W = feat_c.shape
        intrinsic = intrinsic[:, :2, :].clone().contiguous()
        intrinsic = intrinsic.view(B, -1)  # (B, 6)
        src = self.se(src, intrinsic)  # (B, C, H, W)

        # depth feature
        depth_embed = rearrange(self.pos_encoding(src), 'n c h w -> n (h w) c')
        for layer in self.encoder_layers:
            depth_embed = layer(depth_embed, depth_embed)
        depth_embed = rearrange(depth_embed, ' n l c -> n c l').view(B, C, H, W)

        # depth predict
        depth_stem = self.depth_head(src)
        depth_prob = self.depth_prob_conv(depth_stem)
        depth_direct = self.depth_direct_conv(depth_stem)

        depth_score = depth_prob   # 未经过softmax
        depth_prob = depth_prob.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_num)   # (B, H, W, D) --> (B*H*W, D)
        depth_prob_val = self.integral(depth_prob)      # (B*H*W, )
        depth_map_pred = depth_prob_val

        sig_alpha = torch.sigmoid(self.fuse_lambda)
        depth_direct_val = depth_direct.view(-1)      # (B*H*W, )
        depth_pgd_fuse = sig_alpha * depth_direct_val + (1 - sig_alpha) * depth_prob_val
        depth_map_pred = depth_pgd_fuse.view(B, H, W)

        return depth_map_pred, depth_embed, depth_score


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



