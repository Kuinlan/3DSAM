import torch
import torch.nn as nn
from einops.einops import rearrange
from ..utils.position_encoding import pos2posemb3d


class PositionalEncoding3D(nn.Module):
    """Conduct 3D sine positional encoding."""

    def __init__(self, config):
        super().__init__()
        # make in config
        self.depth_max = config['depth_max']
        self.depth_min = config['depth_min']
        self.embed_dims = config['embed_dims'] # 256
        self.position_range = config['position_range']
        self.eps = 1e-5
        
        # define 3d position embedding
        self.position_encoder = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.position_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, depth_map_pred, HW_i, K_cam, T_0to1=None):
        """
        Returns: pos (Tensor): position embedding with shape [N, 2D, h, w]
        """

        # coord remap
        B, H, W = depth_map_pred.shape
        H_in, W_in = HW_i[0], HW_i[1]
        coords_h = torch.arange(H, device=depth_map_pred.device).float() * H_in / H  # (H, )
        coords_w = torch.arange(W, device=depth_map_pred.device).float() * W_in / W  # (W, )

        # (2, W, H)  --> (W, H, 2)    2: (u, v)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).permute(1, 2, 0).contiguous()
        coords = coords.view(1, W, H, 2).repeat(B, 1, 1, 1)  # (B, W, H, 2)
        self.coords2d = coords

        depth_map_pred = depth_map_pred.permute(0, 2, 1).contiguous()  # (B, W, H)
        depth_map_pred = depth_map_pred.unsqueeze(dim=-1)  # (B, W, H, 1)
        coords = coords * torch.maximum(depth_map_pred, torch.ones_like(depth_map_pred) * self.eps)  # (B, W, H, 2)    (du, dv)
        coords = torch.cat([coords, depth_map_pred], dim=-1)     # (B, W, H, 3)   (du, dv, d)
        coords = rearrange(coords, 'n w h c -> n (w h) c')  # (B, L, 3)

        coords3d = K_cam.inverse() @ coords.transpose(2, 1)  # (B, 3, L)
        if T_0to1 is not None:
            coords3d = T_0to1[:, :3, :3] @ coords3d + T_0to1[:, :3, [3]]  # (N, 3, L)

        # reshape
        coords3d = rearrange(coords3d, 'n c (w h) -> n w h c', w=W, h=H)  # (N, W, H, 3)

        # # Normalize to certain scale
        # def norm_coord(coords3d: torch.Tensor):
        #     max = coords3d.view(B, -1, 3).max(dim=1)[0].view(B, 1, 1, 3) 
        #     min = coords3d.view(B, -1, 3).min(dim=1)[0].view(B, 1, 1, 3)
        #     coords3d = (coords3d - min) / (max - min)

        #     return coords3d

        # coords3d = norm_coord(coords3d)

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        # encoding the coordinates
        coords3d = rearrange(coords3d, 'n w h c -> n (h w) c')  # (N, L, 3）
        coords3d = torch.clamp(coords3d, 1e-5, 1-1e-5).float()
        coords3d = torch.log(coords3d / (1 - coords3d))
        coords3d = pos2posemb3d(coords3d)  # (N, L, 3 * embed_dims)
        pos_embedding = self.position_encoder(coords3d) # (N, L, embed_dims)

        return pos_embedding
    
