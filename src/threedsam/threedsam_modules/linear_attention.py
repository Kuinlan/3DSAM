"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
from kornia.utils import create_meshgrid

from ..utils.geometry import get_epipolar_line_std

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


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
    

class One2ManyAttention(Module):
    def __init__(self, dim, nhead, area_width, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps
        self.dim = dim
        self.nhead = nhead
        self.area_width = area_width

    def forward(self, queries, keys, values, epipolar_info, direction, q_mask=None, source_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            source_mask: [N, S]
        Returns:
            queried_values: (N, L, H, C)
        """

        N, L, _, _ = queries.shape
        solved_sample = epipolar_info['solved_sample']

        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        Q_solved = Q[solved_sample]
        K_solved = K[solved_sample]
        values_solved = values[solved_sample]
        
        Q_unsolved = Q[~solved_sample]
        K_unsolved = K[~solved_sample]
        values_unsolved = values[~solved_sample]

        # final result combines two parts splited.
        queried_values = torch.empty((N, L, self.nhead, self.dim), device=queries.device)
        
        if Q_solved.shape[0] > 0:  # epipolar attention
            # Get info for epipolar geometry calculation
            N_solved = solved_sample.sum()
            H_c = epipolar_info['h0c']
            W_c = epipolar_info['w0c']
            K0 = epipolar_info['K0'][solved_sample].clone()
            K1 = epipolar_info['K1'][solved_sample].clone()

            self.coord = create_meshgrid(
                H_c, W_c, False, K0.device
            ).flatten(1, 2)  # [1, L, 2] - <x, y>

            scale = epipolar_info['scale']
            K0 = self.get_scaled_K(K0, scale)
            K1 = self.get_scaled_K(K1, scale)

            R = epipolar_info['R'][solved_sample]
            t = epipolar_info['t'][solved_sample]

            self.max_candidate_num = max(H_c, W_c) * self.area_width

            # pick up candidate information
            if direction == '0to1':
                index, mask = self.get_candidate_index(
                    R, t, K0, K1, self.area_width)  # (N, L, C), (N, L, C) 
            elif direction == '1to0':
                R = R.transpose(1, 2)
                t = -R @ t 
                index, mask = self.get_candidate_index(
                    R, t, K1, K0, self.area_width)
            else:
                raise KeyError

            if q_mask is not None:
                Q_solved = Q_solved * q_mask[:, :, None, None]

            if source_mask is not None:
                K_solved = K_solved * source_mask[:, :, None, None]
                values_solved = values_solved * source_mask[:, :, None, None]
            
            # pick up candidate feature
            N_index = torch.arange(N_solved, device=keys.device)[..., None][..., None]
            K_solved =  K_solved[N_index, index]  # [N'', L, C, H, D]
            values_solved =  values_solved[N_index, index]

            # mask for candidate of lines
            K_solved = K_solved * mask[:, :, :, None, None]
            values_solved = values_solved * mask[:, :, :, None, None]

            v_length = values_solved.size(2)
            values_solved = values_solved / v_length  # prevent fp16 overflow
            KV_solved = torch.einsum("nlchd,nlchv->nlhdv", K_solved, values_solved)  # (S,D)' @ S,V
            Z_solved = 1 / (torch.einsum("nlhd,nlhd->nlh", Q_solved, K_solved.sum(dim=2)) + self.eps)
            queried_values_solved = torch.einsum("nlhd,nlhdv,nlh->nlhv", Q_solved, KV_solved, Z_solved) * v_length
                
            queried_values[solved_sample] = queried_values_solved.contiguous()

        if Q_unsolved.shape[0] > 0:  # regular attention
            # set padded position to zero
            if q_mask is not None:
                Q_unsolved = Q_unsolved * q_mask[:, :, None, None]
            if source_mask is not None:
                K_unsolved = K_unsolved * source_mask[:, :, None, None]
                values_unsolved = values_unsolved * source_mask[:, :, None, None]

            v_length = values_unsolved.size(1)
            values_unsolved = values_unsolved / v_length  # prevent fp16 overflow
            KV_unsolved = torch.einsum("nshd,nshv->nhdv", K_unsolved, values_unsolved)  # (S,D)' @ S,V
            Z_unsolved = 1 / (torch.einsum("nlhd,nhd->nlh", Q_unsolved, K_unsolved.sum(dim=1)) + self.eps)
            queried_values_unsolved = torch.einsum("nlhd,nhdv,nlh->nlhv", Q_unsolved, KV_unsolved, Z_unsolved) * v_length

            queried_values[~solved_sample] = queried_values_unsolved.contiguous()

        return queried_values.contiguous()


    @torch.no_grad()
    def get_mask(self, lines, mode, area_width):
        """
        Args:
            lines (torch.Tensor): [N, L, 3]
            mode  (torch.Tensor): [N, L]

        Return:
            within (torch.Tensor): [N, L, S]
        """  
        S = self.coord.shape[1]
        N, L = lines.shape[0:2]
        coord = self.coord[0]
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
        mask_within_area = self.get_mask(lines, mode, area_width) # [N, L, S] 
        # output = gather_index(mask_within_area, self.max_candidate_num) 
        index, mask = self.gather_index(mask_within_area)

        return index, mask
    

    @torch.no_grad()
    def gather_index(self, x):
        C = self.max_candidate_num
        N, L, _ = x.shape
        indices = torch.arange(L, device=x.device, dtype=torch.int64)[None][None].repeat(N, L, 1)  # [N, L, L]
        true_indices = torch.where(x, indices, torch.tensor(0, device=x.device, dtype=torch.int64))  # [N, L, L]
        true_indices, _ = torch.sort(true_indices, dim=-1)
        indices_output = true_indices[..., -C:]  # [N, L, C]
        mask = indices_output != 0  # [N, L, C]

        return indices_output, mask

        
    @torch.no_grad()
    def get_scaled_K(self, K: torch.Tensor, scale):
        if K.dim() == 2:
            K[:2, :] = K[:2, :] / scale
        elif K.dim() == 3:
            K[:, :2, :] = K[:, :2, :] / scale
        else:
            raise ValueError("Expected tensor of shape: [N, 3, 3] or [3, 3]")

        return K
