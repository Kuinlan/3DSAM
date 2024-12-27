import torch
import torch.nn as nn
import numpy as np
from einops.einops import rearrange

from .perceiver_module import PerceiverLayer
from .perceiver_module import BaseDecoder
from .perceiver_module import sum_list

class PerceiverEmbeddings(nn.Module):
    """Class to create and store the Perceiver IO latent space"""
    def __init__(self, config):
        super().__init__()
        self.shape = (config['latent_num'], config['latent_dim'])
        self.latents = nn.Parameter(torch.randn(self.shape))
        self.freeze = config['freeze']

    def forward(self, batch_size):
        """Forward pass to retrieve the Perceiver IO latent space"""
        if self.training:
            self.latents.requires_grad_(not self.freeze)
        return self.latents.expand(batch_size, -1, -1)

class PerceiverEncoder(nn.Module):
    def __init__(self, config):
        super(PerceiverEncoder, self).__init__()
        self.config = config

        config = self.config['cross_layer'] 
        self.cross_attention = PerceiverLayer(
                is_cross_attention=config['is_cross_attention'],
                cross_attention_shape=config['cross_attention_shape'],
                use_query_residual=config['use_query_residual'],
                qk_channels=config['qk_channels'],
                v_channels=config['v_channels'],
                num_heads=config['num_heads'],
                q_dim=config['d_latents'],
                kv_dim=config['d_model'],
                widening_factor=config['widening_factor'],
                attention_dropout=config['attention_dropout'],
                hidden_activ=config['hidden_activ'],
                use_flash_attention=config['use_flash_attention']
            )
        self_attention_layers = []

        config = self.config['self_layer']        
        for _ in range(config['num_self_attends_per_block']):
            layer = PerceiverLayer(
                is_cross_attention=False,
                cross_attention_shape=config['cross_attention_shape'],
                use_query_residual=config['use_query_residual'],
                qk_channels=config['qk_channels'],
                v_channels=config['v_channels'],
                num_heads=config['num_heads'],
                q_dim=config['d_latents'],
                kv_dim=config['d_latents'],
                widening_factor=config['widening_factor'],
                attention_dropout=config['attention_dropout'],
                hidden_activ=config['hidden_activ'],
                use_flash_attention=config['use_flash_attention'],
            )
            self_attention_layers.append(layer)
        
        self.self_attends = nn.ModuleList(self_attention_layers)

    def forward(self, 
                hidden_state,
                inputs,  # embeddings concated together
                attention_mask=None,
                head_mask=None,
                inputs_mask=None
                ):
        """ Forward pass for the Perceiver encoder"""
        B, L, C = inputs.shape
        all_hidden_state = [hidden_state]
        all_self_attention = []
        all_cross_attention = []
        # For each block
        for j in range(self.config['cross_layer']['num_blocks']): # 1

            cross_output = self.cross_attention(  # [1, 1024, 2048]
                hidden_state,
                attention_mask=attention_mask,
                head_mask=None,
                inputs=inputs,
                inputs_mask=inputs_mask,
            )

            all_cross_attention.append(cross_output['attention'])
            hidden_state = cross_output['mlp']

            all_hidden_state.append(hidden_state)

            # Self-attention always happens, if there are modules
            for i, layer_module in enumerate(self.self_attends):
                self_outputs = layer_module(
                    hidden_state,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i] if head_mask is not None else None,
                )
                hidden_state = self_outputs['mlp']

                all_hidden_state.append(hidden_state)
                all_self_attention.append(self_outputs['attention'])

        return {
            'last_hidden_state': hidden_state,
            'all_hidden_state': all_hidden_state,
            'all_self_attention': all_self_attention,
            'all_cross_attention': all_cross_attention,
        }

class PerceiverDecoder(nn.Module):
    def __init__(self, config):
        super(PerceiverDecoder, self).__init__()
        self.is_variational = config['is_variational']
        self.n_validation_samples = config['n_val_samples']
        self.kld_weight = config['kld_weight']
        self.decoder = BaseDecoder(config)

    def decode(self, encoded, embeddings=None):
        """Decode the data from the latent space"""
        decode_fn = self.multi_decode if self.is_variational else self.single_decode
        return decode_fn(encoded, embeddings)

    def single_decode(self, encoded=None, embeddings=None):
        """Single decode function, taking encoded data and the latent space, and decoding embeddings"""
        output = self.decode_fn(encoded, embeddings, is_variational=True)

        return output

    def decode_fn(self, encoded, embeddings=None, is_variational=False):
        """Decode function, taking encoded data and the latent space, and decoding embeddings"""
        decoded, losses = {}, {}
      
        # Get embedding and latent
        latent = encoded

        # Sample from latent space if it's variational
        if is_variational:
            output_variational = self.sample_from_latent(latent)  # [1, 1024, 2048] >>----sample---->> [1, 1024, 1024]
            losses.update(**{key: val for key, val in output_variational.items() if 'loss' in key})
            latent = output_variational['sampled_latent']

        decoded = self.decoder(  # 使用深度解码器进行解码
            query=embeddings, z=latent) # dict: ['prediction', 'cross_output']
        
        # Return losses, embeddings, and output
        return {
            'losses': losses,
            'embeddings': embeddings,
            'decoded': decoded,  # depth prediction
        }

    def multi_decode(self, encoded, embeddings=None):
        """Decode the data from the latent space multiple times, for statistical analysis"""
        num_evaluations = 1 if self.training else \
            self.n_validation_samples

        losses, decoded = [], {}
        decoded['pred'] = []
        decoded['depth_embed'] = []

        for i in range(num_evaluations):
            output_i = self.single_decode(encoded, embeddings)

            decoded['pred'].append(output_i['decoded']['predictions'])
            decoded['depth_embed'].append(output_i['decoded']['depth_embed'])
            losses.append(output_i['losses'])

        # aggregate
        if not self.training:
            mean, stddev = 0.0, 0.0

            pred = torch.stack(decoded['pred'], 0)
            mean = pred.mean(0)
            stddev = pred.std(0).sum(1, keepdim=True)
            decoded['pred_mean'] = mean
            decoded['pred_stddev'] = stddev

        return {
            'losses': sum_list(losses),
            'embeddings': output_i['embeddings'],
            'decoded': decoded, # Keys: pred, depth_embed
        }


    def sample_from_latent(self, latent):
        """Sample from the latent space, to produce variational inference"""
        n = latent.shape[-1] // 2
        mu, logvar = latent[:, :, :n], latent[:, :, n:]
        logvar = logvar.clamp(max=10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # eps = 0.001 * torch.ones_like(std)
        sampled_latent = eps * std + mu

        output = {
            'sampled_latent': sampled_latent
        }

        if self.training:
            output['kld_loss'] = self.kld_weight * torch.mean(
                - 0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=[1, 2]), dim=0)

        return output

    def forward(self, querys, latent):
        decode_output = self.decode(
            encoded=latent, embeddings=querys
        )

        return decode_output

class Perceiver(nn.Module):
    def __init__(self, config):
        super(Perceiver, self).__init__()
        self.embeddings = PerceiverEmbeddings(config['embedding'])
        self.encoder = PerceiverEncoder(config['encoder'])
        self.decoder = PerceiverDecoder(config['decoder'])
        

    def forward(self, img_embedding0, img_embedding1, data: dict):
        # get informations for camera embedding
        K0 = data['K0']  # [B, 3, 3]
        K1 = data['K1']  # [B, 3, 3]
        rel_depth0 = data['rel_depth0']  # [B, 1, H, W]
        rel_depth1 = data['rel_depth1']
        scale = data['hw0_i'][0] / data['hw0_c'][0]  # 8

        rays0 = self.get_viewdirs(data['hw0_c'], K0, scale=scale)  # [B, 3, H, W]
        rays1 = self.get_viewdirs(data['hw1_c'], K1, scale=scale)

        # add rel_depth information and get cam embeddings
        cam_info0 = torch.cat([rays0, rel_depth0], dim=1)  # [B, 4, H, W]
        cam_info1 = torch.cat([rays1, rel_depth1], dim=1)  # [B, 4, H, W]
        cam_info0 = rearrange(cam_info0, 'n c h w -> n (h w) c')  # [B, L, 4]
        cam_info1 = rearrange(cam_info1, 'n c h w -> n (h w) c')
        cam_embedding0 = generate_fourier_features(cam_info0, 16, [64, 64, 64, 64],
                                                   concat_pos=True, sine_only=False)  # [B, L, 132]
        cam_embedding1 = generate_fourier_features(cam_info1, 16, [64, 64, 64, 64],
                                                   concat_pos=True, sine_only=False)
        img_embedding0 = rearrange(img_embedding0, 'n c h w -> n (h w) c')
        img_embedding1 = rearrange(img_embedding1, 'n c h w -> n (h w) c')

        embeddings0 = torch.cat([img_embedding0, cam_embedding0], dim=-1)  # [B, L, 132+256=388]
        embeddings1 = torch.cat([img_embedding1, cam_embedding1], dim=-1)

        batch_size = embeddings0.shape[0]
        latent0 = self.embeddings(batch_size)
        latent1 = self.embeddings(batch_size)
        
        # encoding
        encoded_data0 = self.encoder(latent0, embeddings0)
        encoded_data1 = self.encoder(latent1, embeddings1)
        latent0 = encoded_data0['last_hidden_state']
        latent1 = encoded_data1['last_hidden_state']

        # decoding
        output0 = self.decoder(cam_embedding0, latent0)
        output1 = self.decoder(cam_embedding1, latent1)
        depth_pred0 = output0['decoded']['pred'][0].squeeze(dim=-1)  # [N, L]
        depth_pred1 = output1['decoded']['pred'][0].squeeze(dim=-1)
        depth_embed0 = output0['decoded']['depth_embed'][0]  # 1024
        depth_embed1 = output1['decoded']['depth_embed'][0]

        if 'kld_loss' in output0['losses']:
            data['kld_loss0'] = output0['losses']['kld_loss']
            data['kld_loss1'] = output1['losses']['kld_loss']

        return depth_pred0, depth_pred1, depth_embed0, depth_embed1

    def norm_pixel_grid(self, grid, hw=None, in_place=False, align_corners=True):
        """Normalize a pixel grid from [W,H] to [-1,+1]."""
        if hw is None:
            hw = grid.shape[-2:]
        if not in_place:
            grid = grid.clone()
        if align_corners:
            grid[:, 0] = 2.0 * grid[:, 0] / (hw[1] - 1) - 1.0
            grid[:, 1] = 2.0 * grid[:, 1] / (hw[0] - 1) - 1.0
        else:
            grid[:, 0] = 2.0 * grid[:, 0] / hw[1] - 1.0
            grid[:, 1] = 2.0 * grid[:, 1] / hw[0] - 1.0
        return grid

    def pixel_grid(self, hw, b=None, with_ones=False, device=None, normalize=False, shake=False, align_corners=True, scale=1):
        """Helper function to generate a pixel grid given [H,W] or [B,H,W]"""
        if isinstance(hw, torch.Tensor):
            b, hw, device = hw.shape[0], hw.shape[-2:], hw.device
        if isinstance(device, torch.Tensor):
            device = device.device
        if align_corners:
            hi, hf = 0, hw[0] - 1
            wi, wf = 0, hw[1] - 1
        else:
            hi, hf = 0.5, hw[0] - 0.5
            wi, wf = 0.5, hw[1] - 0.5
        yy, xx = torch.meshgrid([torch.linspace(hi, hf, hw[0], device=device) * scale,
                                torch.linspace(wi, wf, hw[1], device=device) * scale], indexing='ij')
        if with_ones:
            grid = torch.stack([xx, yy, torch.ones(hw, device=device)], 0)
        else:
            grid = torch.stack([xx, yy], 0)
        if b is not None:
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
        if shake:
            if align_corners:
                rand = torch.rand((b, 2, *hw), device=device)
            else:
                rand = torch.rand((b, 2, *hw), device=device) - 0.5
            grid[:, :2, :, :] += rand
        if normalize:
            grid = self.norm_pixel_grid(grid, align_corners=align_corners)

        return grid

    def reconstruct_depth_map(self, depth, K: torch.Tensor, grid=None, scale=1):
            """Reconstruct 3D pointcloud from z-buffer depth map"""
            if depth is None:
                return None
            b, _, h, w = depth.shape
            if grid is None:
                grid = self.pixel_grid(depth, with_ones=True, device=depth.device, scale=scale).view(b, 3, -1)
            points = torch.matmul(K.inverse(), grid) * depth.view(depth.shape[0], 1, -1)

            return points.view(b, 3, h, w)

    def get_viewdirs(self, hw, K, normalize=None, reflect=False, grid=None, scale=1):
        """Get the view directions of the camera"""

        ones = torch.ones((K.shape[0], 1, *hw), dtype=K.dtype, device=K.device)
        rays = self.reconstruct_depth_map(ones, K, grid=grid, scale=scale)

        if reflect:
            rays[:, 1] = - rays[:, 1]
            rays[:, 2] = - rays[:, 2]

        if normalize is True or normalize == 'unit':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
        if normalize == 'plane':
            rays = rays / torch.norm(rays, dim=1).unsqueeze(1)
            rays = rays / rays[:, [2]]

        return rays

def generate_fourier_features(pos, num_bands=None, max_resolution=None, 
                            concat_pos=True, sine_only=False, freq_sampling='linear'):
    """Generate fourier features from a given set of positions and frequencies"""
    b, n = pos.shape[:2]
    device = pos.device

    if freq_sampling == 'linear':
        min_freq = 1.0
        freq_bands = torch.stack(  
            [torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
                for res in max_resolution], dim=0
        )  # [3, 16]
    elif freq_sampling == 'log':
        freq_bands = torch.stack(
            [2. ** torch.linspace(0., max_res, steps=num_bands)
                for max_res in max_resolution], dim=0
        ).to(pos.device)
    else:
        raise ValueError('Invalid freq_sampling')

    # [l, 3, 1] * [1, 3, 16] -> [l, 3, 16] -> [b, l, 3, 16] -> [b, l, 48]
    per_pos_features = torch.stack([pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0)
    per_pos_features = per_pos_features.reshape(b, n, -1)  # [1, n, 3 * 16]

    if sine_only:
        per_pos_features = torch.sin(np.pi * per_pos_features)
    else:
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )  # [1, n, 48 * 2]

    if concat_pos:
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)

    return per_pos_features