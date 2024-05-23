import torch
import torch.nn as nn
from typing import Dict

from .linear_attention import One2ManyAttention


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
        self.attention = One2ManyAttention(self.dim, self.nhead, self.area_width)
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
            direction (String): '0to1' or '1to0'
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        message = self.attention(query, key, value, epipolar_info, direction, x_mask, source_mask)  # [N, L, H, D]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, D]
        message = self.norm1(message)

        # ffn
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


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

        # multi-head one to many attention or regular cross-attention if R, t are unsolved
        feat0 = self.layer(feat0, feat1, epipolar_info, '0to1', mask0, mask1)
        feat1 = self.layer(feat1, feat0, epipolar_info, '1to0', mask1, mask0)

        return feat0, feat1

