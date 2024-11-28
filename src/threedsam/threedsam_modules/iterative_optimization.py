import torch
import torch.nn as nn

from .structure_extract import StructureExtractor
from ..backbone.resnet_fpn import BasicBlock
from .transformer import LocalFeatureTransformer, AG_RoPe_Transformer
from .geometric_attention import GA_EncoderLayer


class IterativeOptimization(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.layer_assign = config['layer_assign']  
        self.d_model = config['coarse_init']['d_model']
        self.d_struct = config['d_struct']

        assert len(self.layer_assign) == config['n_iter']

        self.layer_num = len(set(self.layer_assign))
        self.anchor_num_min = config['extractor']['anchor_num_min']

        # Modules
        self.struct_extractor = StructureExtractor(config['extractor'])
        self.self_attention = nn.ModuleList([AG_RoPe_Transformer(config['self_attention']) for _ in range(self.layer_num)])
        
        self.cross_attention_layers = nn.ModuleList([GA_EncoderLayer(config['geometric_attention']) for _ in range(self.layer_num)])

        # 3X3 conv
        layer1 = BasicBlock(2*config['d_struct'], 2*config['d_struct'], stride=1)
        layer2 = BasicBlock(2*config['d_struct'], 2*config['d_struct'], stride=1)
        layers = (layer1, layer2)
        self.conv = nn.Sequential(*layers)

        # appearance & structure cross attention
        self.struct_self_attention_layer = AG_RoPe_Transformer(config['struct_self_attention'])
        self.struct_cross_attention_layer = AG_RoPe_Transformer(config['struct_cross_attention'])

        # ffn
        self.mlp = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=False),
        )

        self.merge = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=False),
        )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
    def forward(self, feat_c0, feat_c1, match_mask, n_iter, data):
        layer_idx = self.layer_assign[n_iter]

        data['epipolar_info0'], data['epipolar_info1'] = None, None

        m_struct0, m_struct1 = self.struct_extractor(match_mask, data)  # [N, C, H, W]

        # 3D structure encoding
        m0 = self.mlp(m_struct0.permute(0, 2, 3, 1))  # [N, H, W, C] 
        m1 = self.mlp(m_struct1.permute(0, 2, 3, 1))   

        m0 = self.norm1(m0).permute(0, 3, 1, 2)  # [N, C, H, W]
        m1 = self.norm1(m1).permute(0, 3, 1, 2)

        m0, m1 = self.conv(m0), self.conv(m1) # [N, C, H, W]
        if 'm_struct0' in data:
            m0 = self.merge(torch.cat([m0, data['m_struct0']], dim=1).permute(0, 2, 3, 1))
            m1 = self.merge(torch.cat([m1, data['m_struct1']], dim=1).permute(0, 2, 3, 1))
            m0 = self.norm2(m0).permute(0, 3, 1, 2)
            m1 = self.norm2(m1).permute(0, 3, 1, 2)

        m0, m1 = self.struct_self_attention_layer(m0, m1)
        feat_c0, m0 = self.struct_cross_attention_layer(feat_c0, m0)
        feat_c1, m1 = self.struct_cross_attention_layer(feat_c1, m1)

        data.update({
            'm_struct0': m0,
            'm_struct1': m1
        })

        # mask
        mask_c0 = mask_c1 = None  
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']

        # self-attention
        feat_c0, feat_c1 = self.self_attention[layer_idx](feat_c0, feat_c1, mask_c0, mask_c1)  # [N, C, H, W]

        # geomertric cross-attention
        epipolar_info0, epipolar_info1 = data['epipolar_info0'], data['epipolar_info1'] 

        # disable epipolar cross attention: epipolar_info == None
        feat_c0, update_mask0 = self.cross_attention_layers[layer_idx](feat_c0, feat_c1, None, mask_c0, mask_c1)  # [N, C, H, W]
        feat_c1, update_mask1 = self.cross_attention_layers[layer_idx](feat_c1, feat_c0, None, mask_c1, mask_c0)  # [N, C, H, W]
    
        data.update({
            'update_mask0': update_mask0,
            'update_mask1': update_mask1,
        })

        return feat_c0, feat_c1
