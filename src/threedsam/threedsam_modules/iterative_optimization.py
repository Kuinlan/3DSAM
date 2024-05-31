import torch
import torch.nn as nn

from .structure_extract import StructureExtractor
from .transformer import LocalFeatureTransformer


class IterativeOptimization(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.layer_assign = config['layer_assign']  # [0, 1, 1, 1]
        self.d_model = config['coarse']['d_model']
        self.d_struct = config['d_struct']

        assert len(self.layer_assign) == config['n_iter']

        self.layer_num = len(set(self.layer_assign))

        # Modules
        self.struct_extractor = StructureExtractor(config['extractor'])
        self.self_attention_layers = nn.ModuleList([LocalFeatureTransformer(config['self_attention']) for _ in range(self.layer_num)])
        self.cross_attention_layers = nn.ModuleList([LocalFeatureTransformer(config['epipolar_attention']) for _ in range(self.layer_num)])

        # ffn
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model+self.d_struct, self.d_model+self.d_struct, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(self.d_model+self.d_struct, self.d_model, bias=False),
        )

        self.norm = nn.LayerNorm(self.d_model)
        
    def forward(self, feat_c0, feat_c1, match_mask, n_iter, data):
        epipolar_sample = ~data['non_epipolar']
        layer_idx = self.layer_assign[n_iter]

        # self-attention with 2D RoPE
        feat_c0, feat_c1 = self.self_attention_layers[layer_idx](feat_c0, feat_c1, mask_c0, mask_c1)  # [N, C, H, W]

        # 3D structure information extraction & relative pose estimation
        epi_info0, epi_info1 = None, None
        feat_structured0, feat_structured1 = torch.empty_like(feat_c0), torch.empty_like(feat_c1)  # [N, C, H, W]

        if epipolar_sample.sum() > 0:
            m_struct0, m_struct1 = self.struct_extractor(
                match_mask[epipolar_sample], data
            )  # [N', C, H, W]

            m0 = self.mlp(torch.cat([feat_c0[epipolar_sample], m_struct0], dim=1).permute(0, 2, 3, 1))  # [N', H, W, C] 
            m1 = self.mlp(torch.cat([feat_c1[epipolar_sample], m_struct1], dim=1).permute(0, 2, 3, 1))   

            feat_structured0[epipolar_sample] = feat_c0[epipolar_sample] + m0.permute(0, 3, 1, 2)  # [N', C, H, W]
            feat_structured1[epipolar_sample] = feat_c1[epipolar_sample] + m1.permute(0, 3, 1, 2)
    
            epi_info0 = data['epipolar_info0']
            epi_info1 = data['epipolar_inf01']

        feat_structured0[~epipolar_sample] = feat_c0[~epipolar_sample]
        feat_structured1[~epipolar_sample] = feat_c1[~epipolar_sample]

        mask_c0 = mask_c1 = None  
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'], data['mask1']

        # cross-attention
        feat_c0, feat_c1 = self.cross_attention_layers[layer_idx](feat_structured0, feat_structured1, mask_c0, mask_c1,
                                                                  epipolar_sample, epi_info0, epi_info1)  # [N, C, H, W]

        return feat_c0, feat_c1
