import torch
import torch.nn as nn
from kornia.utils import create_meshgrid
from typing import Dict

from .linear_attention import One2ManyAttention
from ..utils.geometry import get_epipolar_line_std

from variable_gather import gather_index

@torch.no_grad()
def get_mask(coord, lines, mode, area_width):
    """
    Args:
        coord (torch.Tensor): [N, S, 2]
        lines (torch.Tensor): [N, L, 3]
        mode  (torch.Tensor): [N, L]

    Return:
        within (torch.Tensor): [N, L, S]
    """  
    S = coord.shape[1]
    N, L = lines.shape[0:2]
    coord = coord[0]
    lines = lines.flatten(0, 1)  # [N*L, 3]
    mode = mode.flatten(0, 1)  # [N*L,]

    # Ax + By + C = 0 -> y = kx + b
    line_y = -lines[mode][:, [0, 2]] / lines[mode, 1].unsqueeze(-1)
    # Ax + By + C = 0 -> x = (1/k)y + 1/b
    line_x = -lines[~mode][:, [1, 2]] / lines[~mode, 0].unsqueeze(-1)

    coord_y = torch.einsum(
        'l,s->ls', line_y[:, 0], coord[:, 0]
    ) + line_y[:, 1].unsqueeze(dim=-1)  # [N', S]
    coord_x = torch.einsum(
        'l,s->ls', line_x[:, 0], coord[:, 1]
    ) + line_x[:, 1].unsqueeze(dim=-1)  # [N'', S]


    within = torch.empty((N*L, S), dtype=torch.bool, device=lines.device)
    within_y = (
        (coord[None, :, 1] < (coord_y + area_width//2)) & 
        (coord[None, :, 1] > (coord_y - area_width//2))
    )  # [N', S]
    within_x = (
        (coord[None, :, 0] < (coord_x + area_width//2)) &
        (coord[None, :, 0] > (coord_x - area_width//2))
    )
    within[mode, :] = within_y
    within[~mode, :] = within_x
    within = within.view(N, L, S)

    return within  


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, area_width):
        """
        Args:
            d_model (int)
            nhead (int)
            area_width (int)
        """
        super().__init__()

        self.nhead = nhead
        self.dim = d_model // nhead         

        # epipolar geometry info
        self.area_width = area_width

        # multi-head cross attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = One2ManyAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 

    def forward(self, x, source, epipolar_info,
                direction, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            epipolar_info (dict)
            direction (String)
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        
        device = x.device
        # Get info for epipolar geometry calculation
        H = epipolar_info['h0c']
        W = epipolar_info['w0c']
        K0 = epipolar_info['K0'].clone()
        K1 = epipolar_info['K1'].clone()
        scale = epipolar_info['scale']
        R = epipolar_info['R']
        t = epipolar_info['t']

        K0 = get_scaled_K(K0, scale)
        K1 = get_scaled_K(K1, scale)
        self.max_candidate_num = max(H, W) * self.area_width
        self.coord = create_meshgrid(
            H, W, False, K0.device
        ).flatten(1, 2)  # [1, L, 2] - <x, y>
        
        query, key, value = x, source, source

        N = x.shape[0]
        C = self.max_candidate_num

        # projection
        query = self.q_proj(query)
        key = self.k_proj(key) 
        value = self.v_proj(value)

        # pick up candidate information
        if direction == '0to1':
            index, mask = self.get_candidate_index(
                R, t, K0, K1, self.area_width)  
        elif direction == '1to0':
            R = R.transpose(1, 2)
            t = -R @ t 
            index, mask = self.get_candidate_index(
                R, t, K1, K0, self.area_width)
        else:
            raise KeyError

        query = query.view(N, -1, self.dim, self.nhead)  # [N, L, H, D]
        key =  key[torch.arange(N, device=device)[..., None][..., None], 
                   index, :].view(N, -1, C, self.dim, self.nhead)  # [N, L, C, H, D]
        value = value[torch.arange(N, device=device)[..., None][..., None], 
                      index, :].view(N, -1, C, self.dim, self.nhead)
        message = self.attention(query, key, value, x_mask, mask)  # [N, L, H, D]
        message = self.merge(message.view(N, -1, self.nhead*self.dim)) 
        message = self.norm1(message)

        # ffn
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

    @torch.no_grad()
    def get_candidate_index(self, R, t, K0, K1, area_width = 5):
        """
        Args:
            area_width: int
        Return:
            output (List[torch.Tensor]):
                padded_index (torch.Tensor): [N, L, C]
                valid_mask (torch.Tensor): [N, L, C]
        """
        # compute epipolar lines
        coord  = self.coord  # [N, L, 2]

        lines, mode = get_epipolar_line_std(coord, R, t, K0, K1)  # [N, L, 2], [N, L]       
        mask_within_area = get_mask(coord, lines, mode, area_width) 
        output = gather_index(mask_within_area, self.max_candidate_num) 

        return output

@torch.no_grad()
def get_scaled_K(K: torch.Tensor, scale):
    if K.dim() == 2:
        K[:2, :] = K[:2, :] / scale
    elif K.dim() == 3:
        K[:, :2, :] = K[:, :2, :] / scale
    else:
        raise ValueError("Expected tensor of shape: [N, 3, 3] or [3, 3]")

    return K
     
class EpipolarAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config['d_model']

        self.layer = CrossAttention(config['d_model'], config['nhead'], config['area_width'])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        
    def forward(self, feat0, feat1, epipolar_info: Dict,
                mask0=None, mask1=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            epipoloar_info (dict): with keys [] 
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        # multi-head one to many attention
        feat0 = self.layer(feat0, feat1, epipolar_info, '0to1', mask0, mask1)
        feat1 = self.layer(feat1, feat0, epipolar_info, '1to0', mask1, mask0)

        return feat0, feat1