import torch
import torch.nn as nn
from einops.einops import rearrange

from ..utils.anchor_sample import get_anchor
from src.threedsam.utils.geometry import get_point_cloud

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

    def __init__(self):
        super().__init__()
        
    def forward(self, anc_i_ids, anc_j_ids, T_0to1, data):
        
        N = data['rel_depth0'].shape[0]
        H, W = data['hw0_c']
        scale = data['hw0_i'][0] / data['hw0_c'][0]  # 8
        epipolar_info0 = dict(hw0_c = data['hw0_c'],
                             hw1_c = data['hw1_c'], 
                             K0 = data['K0'], 
                             K1 = data['K1'],
                             scale = scale)

        epipolar_info1 = dict(hw0_c = data['hw1_c'],
                             hw1_c = data['hw0_c'],
                             K0 = data['K1'], 
                             K1 = data['K0'],
                             scale = data['hw1_i'][0] / data['hw1_c'][0])

        rel_depth0 = data['rel_depth0'] 
        rel_depth1 = data['rel_depth1'] 
        
        R, t = T_0to1[:, :3, :3], T_0to1[:, :3, [3]]  # [N, 3, 3], [N, 3, 1]
        epipolar_info0['R'] = R # [N, 3, 3]
        epipolar_info0['t'] = t  # [N, 3, 1]

        epipolar_info1['R'] = R.transpose(1, 2)
        epipolar_info1['t'] = -R @ t

        data.update(epipolar_info0 = epipolar_info0,
                    epipolar_info1 = epipolar_info1)

        # 3. compute 3D strucure info 
        pts_3d0 = get_point_cloud(rel_depth0, data['K0'], scale=scale)  # [N, L, 3]
        pts_3d1 = get_point_cloud(rel_depth1, data['K1'], scale=scale)
        pts_3d0 = (R @ pts_3d0.transpose(1, 2) + t).transpose(1, 2)  # align point cloud
        anc_pts_3d0 = torch.gather(pts_3d0, dim=1, index=anc_i_ids[..., None].repeat(1, 1, 3))  # [N, num_anc, 3]
        anc_pts_3d1 = torch.gather(pts_3d1, dim=1, index=anc_j_ids[..., None].repeat(1, 1, 3))

        m_struct0 = pts_3d0.unsqueeze(dim=2) - anc_pts_3d0.unsqueeze(dim=1)  # [N, L, ANCHOR_NUM, 3]
        m_struct1 = pts_3d1.unsqueeze(dim=2) - anc_pts_3d1.unsqueeze(dim=1)

        distance0 = m_struct0.square().sum(dim=-1, keepdim=True).sqrt()  # [N, L, ANCHOR_NUM, 1]
        distance1 = m_struct1.square().sum(dim=-1, keepdim=True).sqrt()  

        m_struct0 = l1_norm(torch.cat([m_struct0, distance0], dim=-1), dim=2) # [N, L, ANCHOR_NUM, 4]
        m_struct1 = l1_norm(torch.cat([m_struct1, distance1], dim=-1), dim=2) 

        m_struct0 = rearrange(m_struct0, 'n l c d -> n l (d c)')
        m_struct1 = rearrange(m_struct1, 'n l c d -> n l (d c)')

        return m_struct0, m_struct1
