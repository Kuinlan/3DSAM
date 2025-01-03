import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from einops.einops import rearrange

from .backbone import build_backbone
from .threedsam_modules import (LocalFeatureTransformer, 
                                FinePreprocess,
                                StructureExtractor,
                                PositionalEncoding3D,
                                DepthGuidedEncoder)
from .utils.position_encoding import PositionEncodingSine, generate_fourier_features
from .utils.coarse_matching import get_coarse_match, get_match_mask
from .utils.fine_matching import FineMatching
from .utils.geometry import estimate_pose_np

INF = 1e9

class ThreeDSAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_anchor = config['num_anchor']
        self.num_bands = config['num_bands']
        self.d_model = config['d_model']
        # matching
        self.temperature = config['match_coarse']['dsmax_temperature']

        # LoFTR Modules for pose estimate
        self.backbone = build_backbone(config)
        self.pos_encoding2d = PositionEncodingSine(config['coarse_init']['d_model']) 
        self.loftr_coarse = LocalFeatureTransformer(config['coarse_init'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config['fine'])
        self.fine_matching = FineMatching()

        # 3DSAM modules
        self.structure_extractor = StructureExtractor()
        self.struct_encoder = nn.Sequential(
            nn.Linear(self.num_anchor*4*self.num_bands, self.d_model),  # 32 * 4 * 16
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )        
        self.rel_depth_embed = nn.Embedding(256, 256)
        self.depth_aware_coarse = DepthGuidedEncoder(config['depth_coarse'])
        self.depth_fine_preprocess = FinePreprocess(config)
        self.depth_aware_fine = LocalFeatureTransformer(config['fine'])
        self.depth_fine_matching = FineMatching()


    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W)
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            } 
        """

        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. LoFTR module 
        T_0to1, feat_c0, feat_c1, \
        anc_i_ids, anc_j_ids, find_anc = self.get_pose(data, feat_c0, feat_c1, feat_f0, feat_f1)

        # 3. extract 3D structure & pos embedding
        m_struct0, m_struct1 = self.structure_extractor(anc_i_ids, anc_j_ids, T_0to1, data)  # [N, L, 128]
        struct_embed0 = generate_fourier_features(m_struct0, 16, [64]*self.num_anchor*4, concat_pos=False, sine_only=True)  # [N, L, 2048]
        struct_embed1 = generate_fourier_features(m_struct1, 16, [64]*self.num_anchor*4, concat_pos=False, sine_only=True)
        struct_embed0 = self.struct_encoder(struct_embed0)
        struct_embed1 = self.struct_encoder(struct_embed1)

        # 4. rel depth embedding
        rel_depth0 = data['rel_depth0']
        rel_depth1 = data['rel_depth1']
        depth_embed0 = self.interpolate_depth_embed(rel_depth0)  # (N, H, W, C)
        depth_embed1 = self.interpolate_depth_embed(rel_depth1) 
        depth_embed0 = rearrange(depth_embed0, 'n c h w -> n (h w) c')  # [N, L ,C]
        depth_embed1 = rearrange(depth_embed1, 'n c h w -> n (h w) c')

        mask_c0, mask_c1 = None, None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        # 5. 3D structure aware & epipolar attention
        epipolar_info0, epipolar_info1 = data['epipolar_info0'], data['epipolar_info1']
        feat_c0, feat_c1 = self.depth_aware_coarse(feat_c0, feat_c1, struct_embed0, struct_embed1, 
                                                   depth_embed0, depth_embed1, mask_c0, mask_c1, 
                                                   epipolar_info0=epipolar_info0, epipolar_info1=epipolar_info1)

        # 6. coarse match
        conf_matrix = self.update_conf_matrix(feat_c0, feat_c1, mask_c0, mask_c1, data) 
        data.update({'conf_matrix': conf_matrix})

        data.update(**get_coarse_match(conf_matrix, self.config['match_coarse'], self.training, data))

        # 7. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.depth_fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:
            feat_f0_unfold, feat_f1_unfold = self.depth_aware_fine(feat_f0_unfold, feat_f1_unfold)
            
        # 8. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)


    def update_conf_matrix(self, feat0, feat1, mask_c0, mask_c1, data):
        feat0, feat1 = map(lambda feat: feat / feat.shape[-1]**.5,
                                [feat0, feat1])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.temperature

        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(), 
                -INF)
        
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2) 
        data.update({'conf_matrix': conf_matrix})

        return conf_matrix

    @torch.no_grad()
    def get_pose(self, data, feat_c0, feat_c1, feat_f0, feat_f1):
        """get pose estimation"""
        N = feat_c0.shape[0]
        device = feat_c0.device
        mask_c0, mask_c1 = None, None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0 = rearrange(self.pos_encoding2d(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding2d(feat_c1), 'n c h w -> n (h w) c')

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        conf_matrix = self.update_conf_matrix(feat_c0, feat_c1, mask_c0, mask_c1, data)

        # coarse matching
        data.update(**get_coarse_match(conf_matrix, self.config['match_coarse'], self.training, data, pick_sample=False))

        # fine-level pre-process
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)  # [M, WW, C]

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data, get_pose=True)

        # estimate relative pose with all the matches
        pixel_thr = 0.5
        conf = 0.99999
        m_bids = data['m_bids'].cpu().numpy()
        pts0 = data['mkpts0_f'].cpu().numpy()
        pts1 = data['mkpts1_f'].cpu().numpy()
        K0 = data['K0'].cpu().numpy()
        K1 = data['K1'].cpu().numpy()
        find_anc = torch.zeros((N), device=device, dtype=torch.bool)
        T_0to1 = torch.zeros((N, 4, 4), device=device)

        m_conf = data['mconf'].clone()
        m_iids = data['m_iids'].clone()
        m_jids = data['m_jids'].clone()
        anc_i_ids = torch.zeros((N, self.num_anchor), dtype=torch.int64, device=device)
        anc_j_ids = torch.zeros((N, self.num_anchor), dtype=torch.int64, device=device)

        # pose estimate
        # extract certain amount of pairs as anchor point
        for bs in range(K0.shape[0]):
            mask = m_bids == bs
            mask_t = torch.from_numpy(mask)
            m_conf_b = m_conf[mask_t]
            i_ids_b, j_ids_b = m_iids[mask_t], m_jids[mask_t] 

            # pick up matches with highest conf
            if (len(m_conf_b) >= self.num_anchor):
                _, indices = m_conf_b.sort()
                indices = indices[:self.num_anchor]
                anc_i_ids[bs] = i_ids_b[indices]
                anc_j_ids[bs] = j_ids_b[indices]
                find_anc[bs] = True
            elif (len(m_conf_b) > 0):
                _, indices = m_conf_b.sort()
                indices = indices.repeat(self.num_anchor)[:self.num_anchor]
                anc_i_ids[bs] = i_ids_b[indices]
                anc_j_ids[bs] = j_ids_b[indices]
                find_anc[bs] = True

            ret = estimate_pose_np(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)
            if ret is None:
                if self.training: # help training
                    T_0to1[bs] = data['T_0to1'][bs]
                else:
                    T_0to1[bs] = torch.eye(4)
            else:
                R, t, inliers = ret
                R = torch.from_numpy(R)
                t = torch.from_numpy(t)
                # check sanity
                if torch.any(torch.isnan(R)) or torch.any(torch.isinf(R)) or torch.any(torch.isnan(t)) or torch.any(torch.isinf(t)):
                    if self.training: # help training
                        T_0to1[bs] = data['T_0to1'][bs]
                    else:
                        T_0to1[bs] = torch.eye(4)
                else:
                    T_0to1[bs][:3, :3] = R
                    T_0to1[bs][:3, 3] = t
            # # For training, 50 percent using ground truth Transformation
            # if self.training:
            #     if np.random.rand() < 0.5: # using Ground Truth
            #         T_0to1[bs] = data['T_0to1'][bs]
            # # before output T, normalize t
            # T_0to1[bs][0:3, 3] = (T_0to1[bs][0:3, 3] / torch.linalg.norm(T_0to1[bs][0:3, 3]))

        
        return T_0to1, feat_c0, feat_c1, anc_i_ids, anc_j_ids, find_anc

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def interpolate_depth_embed(self, depth):
        pos = self.interpolate_1d(depth, self.rel_depth_embed)
        pos = rearrange(pos, 'n h w c -> n c h w')
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings-1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta # [N H W C]
