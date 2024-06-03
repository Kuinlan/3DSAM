import torch
import torch.nn as nn
from kornia.utils import create_meshgrid
from einops.einops import rearrange

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
        
        epipolar_sample = ~data['non_epipolar'] 

        epipolar_info0 = dict(hw0_c = data['hw0_c'],
                             hw1_c = data['hw1_c'], 
                             K0 = data['K0'][epipolar_sample], 
                             K1 = data['K1'][epipolar_sample])

        epipolar_info1 = dict(hw0_c = data['hw1_c'],
                             hw1_c = data['hw0_c'],
                             K0 = data['K1'][epipolar_sample], 
                             K1 = data['K0'][epipolar_sample])

        depthmap_scale = data['hw0_i'][0] // data['hw0_c'][0]  # 8
        pts_3d0 = data['pts_3d0'][epipolar_sample] # [N', L', 3], L' = 640 * 480
        pts_3d1 = data['pts_3d1'][epipolar_sample]
        
        # 1.anchor index padding 
        anchor_i_ids, anchor_j_ids = anchor_index_padding(data, match_mask, 
                                                          self.train_anchor_num, 
                                                          self.pad_num_min, 
                                                          self.training)  # [N, ANCHOR_NUM]
                                                    
        pts_2d0 = torch.stack([anchor_i_ids % data['hw0_c'][1], 
                                anchor_i_ids // data['hw0_c'][1]], dim=-1).to(torch.float32)  # [N', ANCHOR_NUM, 2]
        pts_2d1 = torch.stack([anchor_j_ids % data['hw1_c'][1], 
                                anchor_j_ids // data['hw1_c'][1]], dim=-1).to(torch.float32)
        
        # 2.estimate relative pose using anchor points
        K0 = data['K0'][epipolar_sample].clone()
        K1 = data['K1'][epipolar_sample].clone()
        R, t = estimate_pose(pts_2d0, pts_2d1, K0, K1)

        epipolar_info0['R'] = R  # [N, 3, 3]
        epipolar_info0['t'] = t  # [N, 3, 1]

        epipolar_info1['R'] = R.transpose(1, 2)
        epipolar_info1['t'] = -R @ t

        data.update(epipolar_info0 = epipolar_info0,
                    epipolar_info1 = epipolar_info1)

        # 3. compute 3D relative position to anchor points
        anchor_pts0 = pts_3d0[torch.arange(N).unsqueeze(1), anchor_i_ids, :]  # [N', ANCHOR_NUM, 3] - <x, y, z> 
        anchor_pts1 = pts_3d1[torch.arange(N).unsqueeze(1), anchor_j_ids, :]

        grid_c = create_meshgrid(data['hw0_c'][0], data['hw0_c'][1], False, pts_3d0.device, torch.int64)
        grid_c = (grid_c * depthmap_scale).reshape(-1, 2)  # [L, 2] 
        inds_c = data['hw0_c'][1] * grid_c[:, 1] + grid_c[:, 0]  # [L, ]
        pts_3d0_c = pts_3d0[:, inds_c, :]  # [N', L, 3]
        pts_3d1_c = pts_3d1[:, inds_c, :]
        
        m_struct0 = pts_3d0_c.unsqueeze(dim=2) - anchor_pts0.unsqueeze(dim=1)  # [N', L, ANCHOR_NUM, 3]
        m_struct1 = pts_3d1_c.unsqueeze(dim=2) - anchor_pts1.unsqueeze(dim=1)

        distance0 = m_struct0.square().sum(dim=-1, keepdim=True).sqrt()  # [N', L, ANCHOR_NUM, 1]
        distance1 = m_struct1.square().sum(dim=-1, keepdim=True).sqrt()  

        m_struct0 = l1_norm(torch.cat([m_struct0, distance0], dim=-1), dim=-2) # [N', L, ANCHOR_NUM, 4]
        m_struct1 = l1_norm(torch.cat([m_struct1, distance1], dim=-1), dim=-2) 

        m_struct0 = rearrange(m_struct0, 'n (h w) c d -> n (d c) h w', h=data['hw0_c'][0], w=data['hw0_c'][1])
        m_struct1 = rearrange(m_struct1, 'n (h w) c d -> n (d c) h w', h=data['hw1_c'][0], w=data['hw1_c'][1])

        return m_struct0, m_struct1
    