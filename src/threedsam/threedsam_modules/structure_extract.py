import torch
import torch.nn as nn
from ..utils.index_padding import anchor_index_padding
from kornia.geometry.epipolar import find_essential, decompose_essential_matrix

INF = 1e9

@torch.no_grad()
def estimate_pose(kpts0, kpts1):
    E_mat = find_essential(kpts0, kpts1)
    R_est, _, t_est = decompose_essential_matrix(E_mat)

    return R_est, t_est 


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
        self.dropout_prob = config['dropout_prob']    # 0.8
        self.sample_seed = config['anchor_sampler_seed']

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_color + self.dim_struct, 128),
            nn.ReLU(True),
            nn.Dropout(p = self.dropout_prob),
            nn.Linear(128, 256), 
            nn.ReLU(True),
            nn.Dropout(p = self.dropout_prob),
            nn.Linear(256, self.dim_color)
        )

    def forward(self, feat0, feat1, data):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            conf_matrix (torch.Tensor): [N, L, S]
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
        N, L = feat0.shape
        epipolar_info = dict(h0c = data['hw0_c_16'][0],
                             w0c = data['hw0_c_16'][1],
                             h1c = data['hw1_c_16'][0],
                             w1c = data['hw1_c_16'][1],
                             scale = data['hw0_i'][0] / data['hw0_c_16'][0],
                             K0 = data['K0'], 
                             K1 = data['K1'])
        scale = data['wh0_i'] / data['hw0_c_8']
        pts_3d0 = data['pts_3d0'] # [N, L', 3], L' = 640 * 480
        pts_3d1 = data['pts_3d1']
        mask = data['match_mask']
        
        # 1.get anchor index 
        generator = torch.manual_seed(self.sample_seed)
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        anchor_num = mask.sum(dim=(1, 2))  # [N, ]

        # 2.anchor index padding 
        anchor_i_ids, anchor_j_ids =  anchor_index_padding(data, anchor_num, (i_ids, j_ids),
                                                            self.train_anchor_num, self.pad_num_min,
                                                            generator, self.training)  # [N, ANCHOR_NUM]
        pts_2d0 = torch.stack([anchor_i_ids % data['hw0_c'][1], 
                                anchor_i_ids // data['hw0_c'][1]], dim=-1)  # [N, ANCHOR_NUM, 2]
        pts_2d1 = torch.stack([anchor_j_ids % data['hw1_c'][1], 
                                anchor_j_ids // data['hw1_c'][1]], dim=-1)
        
        # 3.estimate relative pose using anchor points
        R, t = estimate_pose(pts_2d0, pts_2d1)
        epipolar_info['R'] = R
        epipolar_info['t'] = t

        data.update(epipolar_info = epipolar_info)

        # 4. compute 3D relative position to anchor points
        anchor_pts0 = pts_3d0[torch.arange(N).unsqueeze(1), anchor_i_ids*scale, :]  # [N, ANCHOR_NUM, 3] - <x, y, z> 
        anchor_pts1 = pts_3d1[torch.arange(N).unsqueeze(1), anchor_j_ids*scale, :]

        structured_feat0 = pts_3d0.unsqueeze(dim=2) - anchor_pts0.unsqueeze(dim=1)  # [N, L, ANCHOR_NUM, 3]
        structured_feat1 = pts_3d1.unsqueeze(dim=2) - anchor_pts1.unsqueeze(dim=1)

        dist0 = structured_feat0.square().sum(dim=-1, keepdim=True)  # [N, L, ANCHOR_NUM, 1]
        dist1 = structured_feat1.square().sum(dim=-1, keepdim=True)  

        structured_feat0 = l1_norm(torch.cat([structured_feat0, dist0], dim=-1), dim=-2) # [N, L, ANCHOR_NUM, 4]
        structured_feat1 = l1_norm(torch.cat([structured_feat1, dist1], dim=-1), dim=-2) 
        
        structured_feat0 = structured_feat0.transpose(-1, -2).contiguous().view(N, L, -1) 
        structured_feat1 = structured_feat1.transpose(-1, -2).contiguous().view(N, L, -1)

        # 5. feature augmentation
        augmented_feat0 = self.mlp(torch.cat([feat0, structured_feat0], dim=-1));  # [N, L, D_COLOR + D_STRUCT]
        augmented_feat1 = self.mlp(torch.cat([feat1, structured_feat1], dim=-1));  

        return augmented_feat0, augmented_feat1
    