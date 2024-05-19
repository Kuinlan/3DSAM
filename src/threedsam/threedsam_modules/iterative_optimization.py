import torch
import torch.nn as nn

from .structure_extract import StructureExtractor
from .epipolar_attention import EpipolarAttention
from .feature_fusion import FeatureFusion
from .transformer import LocalFeatureTransformer


class IterativeOptimization(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Modules
        self.struct_extractor = StructureExtractor(config['extractor'])
        self.epipolar_attention = EpipolarAttention(config['epipolar_attention'])
        self.self_attention = LocalFeatureTransformer(config['self_attention'])
        self.feat_fusion = FeatureFusion(config['resnetfpn'])

    def forward(self, 
                feats_8, 
                feats_16, 
                feats_32, 
                match_mask, 
                data, last_iter=False):
        feat0_8, feat1_8 = feats_8
        feat0_16, feat1_16 = feats_16
        feat0_32, feat1_32 = feats_32

        # 3D structure information extraction
        feat0_8, feat1_8 = self.struct_extractor(feat0_8, feat1_8, match_mask, data)
        epipolar_info = data['epipolar_info']

        # self-attention 
        feat0_16, feat1_16 = self.self_attention(feat0_16, feat1_16)

        # cross-attention
        feat0_32, feat1_32 = self.epipolar_attention(feat0_32, feat1_32, epipolar_info)

        # feature update between different level
        feat_list0 = self.feat_fusion(feat0_8, feat0_16, feat0_32, data, last_iter)
        feat_list1 = self.feat_fusion(feat1_8, feat1_16, feat1_32, data, last_iter)

        if len(feat_list0) == 3:
            feat0_8, feat0_16, feat0_32 = feat_list0
            feat1_8, feat1_16, feat1_32 = feat_list1

            return [(feat0_8, feat1_8), (feat0_16, feat1_16), (feat0_32, feat1_32)]
        else:
            feat0_8 = feat_list0[0]
            feat1_8 = feat_list1[0]

            return [(feat0_8, feat1_8)]

            







