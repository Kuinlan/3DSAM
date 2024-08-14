import torch
import torch.nn as nn
from einops.einops import rearrange

from ..utils.anchor_sample import get_anchor

INF = 1e9

@torch.no_grad()
def l1_norm(tensor: torch.Tensor, dim: int):
    """L1 normalization
    Args: 
        tensor (torch.Tensor): [N, L, m]
        dim (int)
    """
    norm = tensor.norm(p=1, dim=dim, keepdim=True) 
    normed = tensor / (norm + 1e-6)

    return normed


class StructureExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.train_anchor_num = config['anchor_num']    # 32
        self.train_anchor_thr = config['anchor_thr']    # 0.5
        self.border_rm = config['border_rm']    # 2
        self.dim_color = config['d_color']    # 256
        self.dim_struct = config['d_struct']
        

    def forward(self, match_mask, data):
        """
        Args:
            match_mask (torch.Tensor): [N, L, S]
            data (dict): with keys 
                [pts_3d0 (torch.Tensor): [N, L, 3]
                 pts_3d1 (torch.Tensor): [N, L, 3]]
        Update:
            data (dict): {
                epipolar_info0 (dict)
                epipolar_info1 (dict)
            }
        Returns:
            m_struct0 (torch.Tensor): [N, C, H, W]
            m_struct1 (torch.Tensor): [N, C, H, W]
        """
        N, L, S = match_mask.shape
        
        non_skip_ids = data['non_skip_ids']
        conf_matrix = data['conf_matrix']
        scale = data['hw0_i'][0] / data['hw0_c'][0]  # 8
        epipolar_info0 = dict(hw0_c = data['hw0_c'],
                             hw1_c = data['hw1_c'], 
                             K0 = data['K0'][non_skip_ids], 
                             K1 = data['K1'][non_skip_ids],
                             scale = scale)

        epipolar_info1 = dict(hw0_c = data['hw1_c'],
                             hw1_c = data['hw0_c'],
                             K0 = data['K1'][non_skip_ids], 
                             K1 = data['K0'][non_skip_ids],
                             scale = scale)

        pts_3d0 = data['pts_3d0'][non_skip_ids]  # [N, L, 3]
        pts_3d1 = data['pts_3d1'][non_skip_ids] 
        
        # 1. get coarse match result
        mask_v, all_j_ids = match_mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 2. get anchor points and estimate relative pose   
        anchor_i_ids, anchor_j_ids, R, t = get_anchor(
            b_ids, i_ids, j_ids, mconf,
            self.train_anchor_num, self.training, data
        )  # [N, ANCHOR_NUM, 2]
        
        epipolar_info0['R'] = R  # [N, 3, 3]
        epipolar_info0['t'] = t  # [N, 3, 1]

        epipolar_info1['R'] = R.transpose(1, 2)
        epipolar_info1['t'] = -R @ t

        data.update(epipolar_info0 = epipolar_info0,
                    epipolar_info1 = epipolar_info1)

        # 3. compute 3D relative position to anchor points
        pts_3d0 = (R @ pts_3d0.transpose(1, 2) + t).transpose(1, 2)  # align point cloud
        anchor_pts0 = pts_3d0[torch.arange(N).unsqueeze(1), anchor_i_ids, :]  # [N, ANCHOR_NUM, 3] - <x, y, z> 
        anchor_pts1 = pts_3d1[torch.arange(N).unsqueeze(1), anchor_j_ids, :]

        m_struct0 = pts_3d0.unsqueeze(dim=2) - anchor_pts0.unsqueeze(dim=1)  # [N, L, ANCHOR_NUM, 3]
        m_struct1 = pts_3d1.unsqueeze(dim=2) - anchor_pts1.unsqueeze(dim=1)

        distance0 = m_struct0.square().sum(dim=-1, keepdim=True).sqrt()  # [N, L, ANCHOR_NUM, 1]
        distance1 = m_struct1.square().sum(dim=-1, keepdim=True).sqrt()  

        m_struct0 = l1_norm(torch.cat([m_struct0, distance0], dim=-1), dim=2) # [N, L, ANCHOR_NUM, 4]
        m_struct1 = l1_norm(torch.cat([m_struct1, distance1], dim=-1), dim=2) 

        m_struct0 = rearrange(m_struct0, 'n (h w) c d -> n (d c) h w', 
                              h=data['hw0_c'][0], w=data['hw0_c'][1])
        m_struct1 = rearrange(m_struct1, 'n (h w) c d -> n (d c) h w', 
                              h=data['hw1_c'][0], w=data['hw1_c'][1])

        return m_struct0, m_struct1

    