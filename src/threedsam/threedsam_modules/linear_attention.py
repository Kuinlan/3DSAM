"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
from kornia.utils import create_meshgrid
from einops.einops import rearrange

from ..utils.geometry import get_epipolar_line_std


if hasattr(F, 'scaled_dot_product_attention'):
    FLASH_AVAILABLE = True
    from torch.backends.cuda import sdp_kernel
else:
    FLASH_AVAILABLE = False
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()
    

class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class One2ManyAttention(Module):
    def __init__(self, nhead, dim, area_width=4):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.area_width = area_width

    def forward(self, query, key, value, epipolar_info, data, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            query: [N, H, W, C]
            key: [N, H, W, C]
            value: [N, H, W, C]
            q_mask: [N, L]
            source_mask: [N, S]
        Returns:
            queried_values: (N, L, C)
        """

        N = query.shape[0]

        # Get candidate area info
        c_index, valid_candidate_mask, within_area_mask = self.get_candidate_info(epipolar_info)  # [N, L, C], [N, L, C], [N, L, S]

        # pick up candidate feature
        query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim), [query, key, value])
        N_index = torch.arange(N, device=key.device)[..., None][..., None]
        key = key[N_index, c_index]  # [N', L, C, H, D]
        value = value[N_index, c_index]

        # TODO: add implementation for query mask.
        # prepare mask 
        if kv_mask is not None:
            mask = kv_mask[N_index, c_index] * valid_candidate_mask  # [N, S] -> [N, L, C]
        else:
            mask = valid_candidate_mask

        # full one-to-many attention
        QK = torch.einsum("nlhd,nlchd->nlch", query, key)
        QK.masked_fill_(~mask[..., None], float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        A = torch.nan_to_num(A, nan=0)
        m = torch.einsum("nlch,nlchd->nlhd", A, value)

        return m


    @torch.no_grad()
    def get_candidate_info(self, epipolar_info):
        scale = epipolar_info['agg_scale']

        H0 = epipolar_info['hw0_c'][0] // scale
        W0 = epipolar_info['hw0_c'][1] // scale
        H1 = epipolar_info['hw1_c'][0] // scale
        W1 = epipolar_info['hw1_c'][1] // scale

        K0 = epipolar_info['K0'].clone()
        K1 = epipolar_info['K1'].clone()

        self.coord0 = create_meshgrid(
            H0, W0, False, K0.device
        ).flatten(1, 2)  # [1, L, 2] - <x, y>

        self.coord1 = create_meshgrid(
            H1, W1, False, K1.device
        ).flatten(1, 2)  # [1, L, 2] - <x, y>

        K0 = self.get_scaled_K(K0, scale)
        K1 = self.get_scaled_K(K1, scale)

        R = epipolar_info['R']
        t = epipolar_info['t']

        self.max_candidate_num = max(H1, W1) * self.area_width

        # pick up candidate information
        index, valid_candidate_mask, within_area_mask = self.get_candidate_index(R, t, K0, K1, self.area_width)  # (N, L, C), (N, L, C) 

        within_area_mask = rearrange(within_area_mask, 'n l (h w) -> n l h w', h=H1, w=W1)  # [N, L, H, W]

        return index, valid_candidate_mask, within_area_mask


    @torch.no_grad()
    def get_candidate_index(self, R, t, K0, K1, area_width = 5):
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
        within_area_mask = self.get_mask(lines, mode, area_width) # [N, L, S] 
        index, valid_candidate_mask = self.gather_index(within_area_mask)  # [N, L, C]

        return index, valid_candidate_mask, within_area_mask


    @torch.no_grad()
    def get_mask(self, lines, mode, area_width):
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
        

    @torch.no_grad()
    def gather_index(self, x):
        C = self.max_candidate_num
        N, L, _ = x.shape
        indices = torch.arange(L, device=x.device, dtype=torch.int64)[None][None].repeat(N, L, 1)  # [N, L, L]
        true_indices = torch.where(x, indices, torch.tensor(0, device=x.device, dtype=torch.int64))  # [N, L, L]
        true_indices, _ = torch.sort(true_indices, dim=-1)
        indices_output = true_indices[..., -C:]  # [N, L, C]
        valid_mask = indices_output != 0  # [N, L, C]

        return indices_output, valid_mask

        
    @torch.no_grad()
    def get_scaled_K(self, K: torch.Tensor, scale):
        if K.dim() == 2:
            K[:2, :] = K[:2, :] / scale
        elif K.dim() == 3:
            K[:, :2, :] = K[:, :2, :] / scale
        else:
            raise ValueError("Expected tensor of shape: [N, 3, 3] or [3, 3]")

        return K

    def update_mask(self, mask, data):
        NotImplemented



def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = x_mask[0].sum(-2)[0], x_mask[0].sum(-1)[0], source_mask[0].sum(-2)[0], source_mask[0].sum(-1)[0]
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0

def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, L, H, D = m.size()
    m = m.view(bs, mask_h0, mask_w0, H, D)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2)-mask_h0, x_mask.size(-1), H, D, device=m.device, dtype=m.dtype)], dim=1)
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat([m, torch.zeros(m.size(0), x_mask.size(-2), x_mask.size(-1)-mask_w0, H, D, device=m.device, dtype=m.dtype)], dim=2)
    return m

class Attention(Module):
    def __init__(self, no_flash=False, nhead=8, dim=256, fp32=False):
        super().__init__()
        self.flash = FLASH_AVAILABLE and not no_flash
        self.nhead = nhead
        self.dim = dim
        self.fp32 = fp32
        
    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert q_mask is None and kv_mask is None, "Not support generalized attention mask yet."
        if self.flash and not self.fp32:
            args = [x.contiguous() for x in [query, key, value]]
            with sdp_kernel(enable_math= False, enable_flash= True, enable_mem_efficient= False):
                out = F.scaled_dot_product_attention(*args)
        elif self.flash:
            args = [x.contiguous() for x in [query, key, value]]
            out = F.scaled_dot_product_attention(*args)
        else:
            QK = torch.einsum("nlhd,nshd->nlsh", query, key)
    
            # Compute the attention and the weighted average
            softmax_temp = 1. / query.size(3)**.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=2)

            out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(query, key, value, q_mask, kv_mask)

        if self.flash:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n nhead (h w) d', nhead=self.nhead, d=self.dim), [query, key, value])
        else:
            query, key, value = map(lambda x: rearrange(x, 'n h w (nhead d) -> n (h w) nhead d', nhead=self.nhead, d=self.dim), [query, key, value])

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if self.flash:
            m = rearrange(m, 'n nhead L d -> n L nhead d', nhead=self.nhead, d=self.dim)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)
        
        return m
    
    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            if FLASH_AVAILABLE: # pytorch scaled_dot_product_attention
                queries: [N, H, L, D]
                keys: [N, H, S, D]
                values: [N, H, S, D]
            else:
                queries: [N, L, H, D]
                keys: [N, S, H, D]
                values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:            
            m = self._forward(query, key, value, q_mask=q_mask, kv_mask=kv_mask)
        else: # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(self._forward(query[i:i+1], key[i:i+1], value[i:i+1], q_mask=q_mask[i:i+1], kv_mask=kv_mask[i:i+1]))
            m = torch.cat(m_list, dim=0)
        return m