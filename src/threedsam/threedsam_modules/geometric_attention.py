import torch
import torch.nn as nn
from einops.einops import rearrange
from kornia import create_meshgrid

from ..utils.geometry import get_epipolar_line_std, get_scaled_K
from ..utils.position_encoding import RoPEPositionEncodingSine
from src.threedsam.threedsam_modules.linear_attention import Attention

class EpipolarAttention(nn.Module):
    def __init__(self, nhead=8, dim=256, area_width=10):
        super().__init__()
        self.nhead = nhead
        self.dim = dim
        self.area_width = area_width
        
    def forward(self, query, key, value, geometry_info=None, q_mask=None, kv_mask=None):
        """ 
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        
        # masking
        attention_mask = self.get_mask(geometry_info)  # [N, L, S]
        if kv_mask is not None:
            qk_mask = attention_mask * (q_mask[:, :, None, None] * kv_mask[:, None, :, None])
        else:
            qk_mask = attention_mask
        QK = torch.masked_fill(QK, ~qk_mask[..., None], float('-inf'))
        
        # Compute the attention and the weighted average
        softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        A = torch.nan_to_num(A, nan=0)

        out = torch.einsum("nlsh,nshd->nlhd", A, value)

        return out.contiguous()
    
    @torch.no_grad()
    def get_mask(self, epipolar_info):
        # agg_scale = epipolar_info['agg_scale'] if 'agg_scale' in epipolar_info.keys() else 1
        scale = epipolar_info['scale'] 

        R = epipolar_info['R']
        t = epipolar_info['t']

        H0 = epipolar_info['hw0_c'][0]
        W0 = epipolar_info['hw0_c'][1]
        H1 = epipolar_info['hw1_c'][0]
        W1 = epipolar_info['hw1_c'][1]

        K0 = epipolar_info['K0'].clone()
        K1 = epipolar_info['K1'].clone()
        K0 = get_scaled_K(K0, scale)
        K1 = get_scaled_K(K1, scale)

        self.coord0 = create_meshgrid(H0, W0, False, K0.device).flatten(1, 2)  # [1, L, 2] - <x, y>
        self.coord1 = create_meshgrid(H1, W1, False, K1.device).flatten(1, 2)  # [1, L, 2] - <x, y>
        self.max_candidate_num = max(H1, W1) * self.area_width
        attention_mask = self.get_epipolar_mask(R, t, K0, K1, self.area_width)  # (N, L, C)

        # scale = scale * agg_scale
        # H0, W0, H1, W1 = map(lambda x: x // agg_scale, [H0, W0, H1, W1])
        # self.coord0 = create_meshgrid(H0, W0, False, K0.device).flatten(1, 2)  # [1, L, 2] - <x, y>
        # self.coord1 = create_meshgrid(H1, W1, False, K1.device).flatten(1, 2)  # [1, L, 2] - <x, y>
        # K0 = get_scaled_K(K0, scale)
        # K1 = get_scaled_K(K1, scale)
        # attention_mask = self.get_epipolar_mask(R, t, K0, K1, self.area_width // agg_scale)

        return attention_mask

    @torch.no_grad()
    def get_epipolar_mask(self, R, t, K0, K1, area_width = 10):
        """
        Args:
            area_width: int
        Return:
            output (List[torch.Tensor]):
                index (torch.Tensor): [N, L, C]
                valid_candidate_mask (torch.Tensor): [N, L, C]
                within_area_mask (torch.Tensor): [N, L, H, W]
        """
        # compute epipolar lines
        coord  = self.coord0  # [N, L, 2]

        lines, mode = get_epipolar_line_std(coord, R, t, K0, K1)  # [N, L, 2], [N, L]       
        within_area_mask = self.get_candidate_mask(lines, mode, area_width) # [N, L, S] 

        return within_area_mask


    @torch.no_grad()
    def get_candidate_mask(self, lines, mode, area_width):
        """
        Args:
            lines (torch.Tensor): [N, L, 3]
            mode  (torch.Tensor): [N, L]

        Return:
            within (torch.Tensor): [N, L, S]
        """  
        S = self.coord1.shape[1]
        N, L = lines.shape[0:2]
        coord = self.coord1[0]  # [S, 2]
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
            (coord[None, :, 1] < (coord_y + area_width/2.0)) & 
            (coord[None, :, 1] > (coord_y - area_width/2.0))
        )  # [N', S]
        within_x = (
            (coord[None, :, 0] < (coord_x + area_width/2.0)) &
            (coord[None, :, 0] > (coord_x - area_width/2.0))
        )

        within[mode, :] = within_y
        within[~mode, :] = within_x
        within = within.view(N, L, S)

        return within  

class GA_EncoderLayer(nn.Module):
    def __init__(self, config):
        super(GA_EncoderLayer, self).__init__()

        d_model = config['d_model']
        self.fp32 = not (config['mp'] or config['half'])
        self.nhead = config['nhead']
        self.dim = d_model // self.nhead
        self.linear = config['linear_attention']

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=self.agg_size0, padding=0, stride=self.agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=config['npe'], ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        

        self.attention = Attention(config['no_flash'], config['nhead'], self.dim, self.fp32, config['linear_attention'])
        self.geometric_attention = EpipolarAttention(self.nhead, self.dim, config['area_width']) 

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, epipolar_info=None, x_mask=None, source_mask=None, data=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        assert x_mask is None and source_mask is None
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        update_mask = None
        if epipolar_info is None:
            m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        else:
            epipolar_info['agg_scale'] = self.agg_size0
            m, update_mask = self.geometric_attention(query, key, value, epipolar_info, x_mask, source_mask)

        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m, update_mask

