import copy
import torch
import torch.nn as nn
from einops.einops import rearrange
from flash_attn import flash_attn_varlen_func, flash_attn_func

from .attention import Attention, LinearAttention, FullAttention
from ..utils.position_encoding import RoPEPositionEncodingSine


class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 linear=False,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        

        self.attention = Attention(no_flash, self.nhead, self.dim, fp32, linear)

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

    def forward(self, x, source, x_mask=None, source_mask=None, data=None):
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
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)

        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class AG_RoPe_Transformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(AG_RoPe_Transformer, self).__init__()
        
        self.fp32 = not (config['mp'] or config['half'])
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.agg_size0, self.agg_size1 = config['agg_size0'], config['agg_size1']
        self.rope = config['rope']

        self_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], config['rope'], config['npe'], config['linear_attention'], self.fp32)
        cross_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], False, config['npe'], config['linear_attention'], self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' \
                                     else copy.deepcopy(cross_layer) for _ in self.layer_names])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]
        
        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0], mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None

            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        if feature_cropped:
            # padding feature
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)], dim=-2)
            elif mask_w0 != mask_W0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=feat0.device, dtype=feat0.dtype)], dim=-1)

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)], dim=-2)
            elif mask_w1 != mask_W1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=feat1.device, dtype=feat1.dtype)], dim=-1)

        return feat0, feat1
        

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask) # temperature = 1 / sqrt(dim) 
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'])  # linear attention
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class TransformerEncoderLayerGeneral(nn.Module):
    
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayerGeneral, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        # MultiHead Linear Attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)
        
        self.FFN = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False)
        )
        
        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, source, query_embed=None, key_embed=None, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        # Embed PE
        if query_embed is not None and key_embed is not None:
            query = x + query_embed
            key = source + key_embed
            value = source
        else:
            query, key, value = x, source, source
        
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim) # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim) # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # temperature = 1 / sqrt(dim) 
        message = self.merge(message.view(bs, -1, self.nhead * self.dim)) # [N, L, C]
        message = self.norm1(message)
        
        # feed-forward network
        message = self.FFN(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        
        return x + message

class DepthGuidedEncoder(nn.Module):
    """A Local Feature Transformer Guilded with depth information."""
    def __init__(self, config):
        super(DepthGuidedEncoder, self).__init__()
        
        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.n_visual_layers = config['num_visual_layers']
        self.n_depth_layers = config['num_depth_layers']
        # setup encoder layers
        encoder_layer = TransformerEncoderLayerGeneral(self.d_model, self.nhead)
        self.depth_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.n_depth_layers)])
        self.selfAttn_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.n_visual_layers)])
        self.crossAttn_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.n_visual_layers)])
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, feat0, feat1, pos_embed0, pos_embed1, depth_embed0=None, depth_embed1=None, mask0=None, mask1=None, ):
        """Transform visual feature together with depth features
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            pos_embed0 (torch.Tensor): [N, L, C]
            pos_embed1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"
        for depth_layer, selfAttn_layer, crossAttn_layer in zip(self.depth_layers, self.selfAttn_layers, self.crossAttn_layers):
            # Fetch depth information into features
            if depth_embed0 is not None:
                feat0 = depth_layer(feat0, depth_embed0, None, None, mask0, mask0)
                feat1 = depth_layer(feat1, depth_embed1, None, None, mask1, mask1)
            # First self attention with visual information
            feat0 = selfAttn_layer(feat0, feat0, pos_embed0, pos_embed0, mask0, mask0)
            feat1 = selfAttn_layer(feat1, feat1, pos_embed1, pos_embed1, mask1, mask1)
            # Finally cross attention between two image features
            feat0_after = crossAttn_layer(feat0, feat1, pos_embed0, pos_embed1, mask0, mask1)
            feat1_after = crossAttn_layer(feat1, feat0, pos_embed1, pos_embed0, mask1, mask0)
            # assign new value
            feat0 = feat0_after
            feat1 = feat1_after
        return feat0, feat1