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
        self.feat_fusion = FeatureFusion(config['resnet_fpn'])

    def forward(self, feats_8, feats_16, feats_32, data):
        feat0_8, feat1_8 = feats_8
        feat0_16, feat1_16 = feats_16
        feat0_32, feat1_32 = feats_32

        # 3D information extraction
        feat0_8, feat1_8 = self.struct_extractor(feat0_8, feat1_8, data)
        epipolar_info = data['epipolar_info']

        # cross attention 
        feat0_16, feat1_16 = self.epipolar_attention(feat0_16, feat1_16, epipolar_info)

        # self attention
        feat0_32, feat1_32 = self.self_attention(feat0_32, feat1_32)

        # feature update between different level
        feat0_8, feat0_16, feat0_32 = self.feat_fusion(feat0_8, feat0_16, feat0_32) 
        feat1_8, feat1_16, feat1_32 = self.feat_fusion(feat1_8, feat1_16, feat1_32) 

        return (feat0_8, feat1_8), (feat0_16, feat1_16), (feat0_32, feat1_32)





