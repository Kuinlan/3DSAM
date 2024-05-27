import torch
import torch.nn as nn
import numpy as np
import cv2 
from kornia.utils import create_meshgrid

from ..utils.index_padding import anchor_index_padding
from ..utils.geometry import estimate_pose

INF = 1e9


@torch.no_grad()
def l1_norm(tensor: torch.Tensor, dim: int):
    """L1 normalization
    Args: 
        tensor (torch.Tensor): [N, L, m]
        dim (int)
    """
    norm = tensor.norm(p=1, dim=dim, keepdim=True) 
    normed = tensor / norm

    return normed


class StructureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.train_anchor_num = config['anchor_num']    # 32
        self.train_anchor_thr = config['anchor_thr']    # 0.5
        self.pad_num_min = config['train_pad_anchor_num_min']  # 8
        self.border_rm = config['border_rm']    # 2
        self.dim_color = config['d_color']    # 256
        self.dim_struct = config['d_struct']    # 128

        # TODO: Using KAN to replace mlp
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_color+self.dim_struct, self.dim_color+self.dim_struct),
            nn.ReLU(True),
            nn.Linear(self.dim_color+self.dim_struct, self.dim_color), 
        )


    def forward(self, feat0, feat1, match_mask, data):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            match_mask (torch.Tensor): [N, L, S]
            data (dict): with keys 
                [pts_3d0 (torch.Tensor): [N, L, 3]
                 pts_3d1 (torch.Tensor): [N, L, 3]]
        Update:
            data (dict): {
                'R_est' (torch.Tensor): (N, 3, 3),
                't_est' (torch.Tensor): (N, 3, 1)
            }
        Returns:
            augmented_feat0 (torch.Tensor): [N, L, C]
            augmented_feat1 (torch.Tensor): [N, S, C]
        """
        N, L, _ = feat0.shape
        _device = feat0.device

        skip_sample = data['skip_sample'] if self.training else torch.tensor([False], dtype=torch.bool, device=_device)

        epipolar_info = dict(h0c = data['hw0_c_32'][0],
                             w0c = data['hw0_c_32'][1],
                             h1c = data['hw1_c_32'][0],
                             w1c = data['hw1_c_32'][1],
                             scale = data['hw0_i'][0] / data['hw0_c_32'][0],
                             K0 = data['K0'][~skip_sample], 
                             K1 = data['K1'][~skip_sample])

        depthmap_scale = data['hw0_i'][0] // data['hw0_c_8'][0]  # 8
        pts_3d0 = data['pts_3d0'][~skip_sample] # [N', L', 3], L' = 640 * 480
        pts_3d1 = data['pts_3d1'][~skip_sample]

        mask = match_mask

        # 1.anchor index padding 
        anchor_i_ids, anchor_j_ids = anchor_index_padding(data, mask, 
                                                          self.train_anchor_num, 
                                                          self.pad_num_min, 
                                                          self.training)  # [N, ANCHOR_NUM]
                                                    
        pts_2d0 = torch.stack([anchor_i_ids % data['hw0_c_8'][1], 
                                anchor_i_ids // data['hw0_c_8'][1]], dim=-1).to(torch.float32)  # [N, ANCHOR_NUM, 2]
        pts_2d1 = torch.stack([anchor_j_ids % data['hw1_c_8'][1], 
                                anchor_j_ids // data['hw1_c_8'][1]], dim=-1).to(torch.float32)
        
        # 2.estimate relative pose using anchor points
        K0 = data['K0'][~skip_sample].clone()
        K1 = data['K1'][~skip_sample].clone()
        R, t = estimate_pose(pts_2d0, pts_2d1, K0, K1)

        epipolar_info['solved_sample'] = torch.ones((N, ), dtype=torch.bool, device=_device)
        epipolar_info['R'] = R
        epipolar_info['t'] = t

        data.update(epipolar_info = epipolar_info)

        # 3. compute 3D relative position to anchor points
        anchor_pts0 = pts_3d0[torch.arange(N).unsqueeze(1), anchor_i_ids, :]  # [N', ANCHOR_NUM, 3] - <x, y, z> 
        anchor_pts1 = pts_3d1[torch.arange(N).unsqueeze(1), anchor_j_ids, :]

        grid_c = create_meshgrid(data['hw0_c_8'][0], data['hw0_c_8'][1], False, pts_3d0.device, torch.int64)
        grid_c = (grid_c * depthmap_scale).reshape(-1, 2)  # [L, 2] 
        inds_c = data['hw0_c_8'][1] * grid_c[:, 1] + grid_c[:, 0]  # [L, ]
        pts_3d0_c = pts_3d0[:, inds_c, :]  # [N', L, 3]
        pts_3d1_c = pts_3d1[:, inds_c, :]
        # pts_3d0_c = pts_3d0.view(N, 480, 640, 3)[:, ::scale, ::scale, :].flatten(1, 2)
        # pts_3d1_c = pts_3d1.view(N, 480, 640, 3)[:, ::scale, ::scale, :].flatten(1, 2)
        
        structured_info0 = pts_3d0_c.unsqueeze(dim=2) - anchor_pts0.unsqueeze(dim=1)  # [N', L, ANCHOR_NUM, 3]
        structured_info1 = pts_3d1_c.unsqueeze(dim=2) - anchor_pts1.unsqueeze(dim=1)

        distance0 = structured_info0.square().sum(dim=-1, keepdim=True)  # [N', L, ANCHOR_NUM, 1]
        distance1 = structured_info1.square().sum(dim=-1, keepdim=True)  

        structured_info0 = l1_norm(torch.cat([structured_info0, distance0], dim=-1), dim=-2) # [N', L, ANCHOR_NUM, 4]
        structured_info1 = l1_norm(torch.cat([structured_info1, distance1], dim=-1), dim=-2) 
        
        structured_info0 = structured_info0.transpose(-1, -2).contiguous().view(N, L, -1)  # [N', L, 4 * ANCHOR_NUM]
        structured_info1 = structured_info1.transpose(-1, -2).contiguous().view(N, L, -1)

        # 4. feature augmentation
        structured_feat0 = self.mlp(torch.cat([feat0, structured_info0], dim=-1));  # [N', L, D_COLOR + D_STRUCT]
        structured_feat1 = self.mlp(torch.cat([feat1, structured_info1], dim=-1));  

        structured_feat0 = feat0 + structured_feat0 
        structured_feat1 = feat1 + structured_feat1 

        return structured_feat0, structured_feat1
    