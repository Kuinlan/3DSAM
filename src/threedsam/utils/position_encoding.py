import math
import torch
from torch import nn
import numpy as np


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

        
class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1) # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1) # [W, 1]
        
        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[3] # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W
        
        div_term = torch.exp(torch.arange(0, d_model//4, 1).float() * (-math.log(10000.0) / (d_model//4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]        

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, :x.size(1), :x.size(2), :]) + (self.rotate_half(x) * self.sin[:, :x.size(1), :x.size(2), :])
    
    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))  # [N, H, W, C//2, 2]
        x1, x2 = x.unbind(dim=-1)  # [N, H, W, C//2], [N, H, W, C//2] 
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    Args:
        pos: (..., 3)
        num_pos_feats:
        temperature:
    Returns:
        posemb: (..., num_feats * 3)
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)     # (num_feats, )
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)   # (num_feats, )   [10000^(0/128), 10000^(0/128), 10000^(2/128), 10000^(2/128), ...]
    pos_x = pos[..., 0, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_x/10000^(0/128), pos_x/10000^(0/128), pos_x/10000^(2/128), pos_x/10000^(2/128), ...]
    pos_y = pos[..., 1, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_y/10000^(0/128), pos_y/10000^(0/128), pos_y/10000^(2/128), pos_y/10000^(2/128), ...]
    pos_z = pos[..., 2, None] / dim_t   # (N_query, num_feats)      num_feats:  [pos_z/10000^(0/128), pos_z/10000^(0/128), pos_z/10000^(2/128), pos_z/10000^(2/128), ...]

    # (N, L, num_feats/2, 2) --> (N, L, num_feats)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_x/10000^(0/128)), cos(pos_x/10000^(0/128)), sin(pos_x/10000^(2/128)), cos(pos_x/10000^(2/128)), ...]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_y/10000^(0/128)), cos(pos_y/10000^(0/128)), sin(pos_y/10000^(2/128)), cos(pos_y/10000^(2/128)), ...]
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)       # num_feats:  [sin(pos_z/10000^(0/128)), cos(pos_z/10000^(0/128)), sin(pos_z/10000^(2/128)), cos(pos_z/10000^(2/128)), ...]
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)   # (N_query, num_feats * 3)

    return posemb
    
def generate_fourier_features(pos, num_bands=None, max_resolution=None, 
                            concat_pos=True, sine_only=False, freq_sampling='linear'):
    """Generate fourier features from a given set of positions and frequencies"""
    b, l = pos.shape[:2]
    device = pos.device

    if freq_sampling == 'linear':
        min_freq = 1.0
        freq_bands = torch.stack(  
            [torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
                for res in max_resolution], dim=0
        )  # [128, 16]
    elif freq_sampling == 'log':
        freq_bands = torch.stack(
            [2. ** torch.linspace(0., max_res, steps=num_bands)
                for max_res in max_resolution], dim=0
        ).to(pos.device)
    else:
        raise ValueError('Invalid freq_sampling')

    # [l, 128, 1] * [1, 128, 16] -> [l, 128, 16] -> [b, l, 128, 16] -> [b, l, 256 * 8]
    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, l, -1)  # [b, l, 256 * 8]

    if sine_only:
        per_pos_features = torch.sin(np.pi * per_pos_features)
    else:
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )  # [1, n, 48 * 2]

    if concat_pos:
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features