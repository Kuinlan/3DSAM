import torch
import torch.nn as nn

from .structure_extract import StructureExtractor
from .transformer import LocalFeatureTransformer
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
        self.self_attention = nn.ModuleList([LocalFeatureTransformer(config['self_attention']) for _ in range(self.layer_num)])
        
        self.cross_attention_layers = nn.ModuleList([GA_EncoderLayer(config['geometric_attention']) for _ in range(self.layer_num)])

        # ffn
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model+self.d_struct, self.d_model+self.d_struct, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.d_model+self.d_struct, self.d_model, bias=False),
        )

        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, feat_c0, feat_c1, match_mask, n_iter, non_skip_ids, data):
        layer_idx = self.layer_assign[n_iter]

        data['epipolar_info0'], data['epipolar_info1'] = None, None

        m_struct0, m_struct1 = self.struct_extractor(match_mask, non_skip_ids, data)  # [N, C, H, W]

        m0 = self.mlp(torch.cat([feat_c0, m_struct0], dim=1).permute(0, 2, 3, 1))  # [N, H, W, C] 
        m1 = self.mlp(torch.cat([feat_c1, m_struct1], dim=1).permute(0, 2, 3, 1))   

        m0 = self.norm(m0).permute(0, 3, 1, 2)  # [N, C, H, W]
        m1 = self.norm(m1).permute(0, 3, 1, 2)

        feat_structured0 = feat_c0 + m0
        feat_structured1 = feat_c1 + m1

        # mask
        mask_c0 = mask_c1 = None  
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']

        # self-attention
        feat_c0, feat_c1 = self.self_attention[layer_idx](feat_structured0, feat_structured1, mask_c0, mask_c1)  # [N, C, H, W]

        # geomertric cross-attention
        epipolar_info0, epipolar_info1 = data['epipolar_info0'], data['epipolar_info1'] 

        # disable epipolar cross attention
        feat_c0, update_mask0 = self.cross_attention_layers[layer_idx](feat_c0, feat_c1, None, mask_c0, mask_c1)  # [N, C, H, W]
        feat_c1, update_mask1 = self.cross_attention_layers[layer_idx](feat_c1, feat_c0, None, mask_c1, mask_c0)  # [N, C, H, W]
    
        data.update({
            'update_mask0': update_mask0,
            'update_mask1': update_mask1,
        })

        return feat_c0, feat_c1
