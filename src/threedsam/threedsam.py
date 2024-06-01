import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .threedsam_modules import (LocalFeatureTransformer, 
                                FinePreprocess,
                                IterativeOptimization)
from .utils.coarse_matching import get_coarse_match, get_match_mask
from .utils.fine_matching import FineMatching

INF = 1e9

class ThreeDSAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.init_attention = LocalFeatureTransformer(config['coarse'])
        self.iterative_optimization = IterativeOptimization(config)
        self.fine_preprocess = FinePreprocess(config)
        self.threedsam_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

        self.iter_num = config['n_iter']
        self.update_weight = nn.Parameter(torch.tensor([0.8 for _ in range(self.iter_num)]))

        self.temperature = config['match_coarse']['dsmax_temperature']

        # for getting anchor points
        self.thr = config['extractor']['anchor_thr']
        self.border_rm = config['extractor']['border_rm']

        # for interference
        self.anchor_num_min = config['extractor']['anchor_num_min']

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

        # 2. init attention and matching
        mask_c0, mask_c1 = None, None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        
        feat_c0, feat_c1 = self.init_attention(feat_c0, feat_c1, None, mask_c0, mask_c1)

        def get_conf_matrix(feat0, feat1, data):
            feat0 = rearrange(feat0, 'n c h w -> n (h w) c')
            feat1 = rearrange(feat1, 'n c h w -> n (h w) c')
            
            # normalize
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

        def update_conf_matrix(feat0, feat1, weight, pre_conf_matrix, data):
            feat0 = rearrange(feat0, 'n c h w -> n (h w) c')
            feat1 = rearrange(feat1, 'n c h w -> n (h w) c')

            feat0, feat1 = map(lambda feat: feat / feat.shape[-1]**.5,
                                   [feat0, feat1])
            sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.temperature

            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(), 
                    -INF)
            conf_matrix_cur = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2) 

            conf_matrix = weight * conf_matrix_cur + (1 - weight) * pre_conf_matrix

            data.update({'conf_matrix': conf_matrix})

            return conf_matrix

        # 2. initialize confidence matrix
        conf_matrix = get_conf_matrix(feat_c0, feat_c1, data)

        # 3. iterative optimization
        for n_iter in range(self.iter_num):
            match_mask = get_match_mask(conf_matrix, self.thr, self.border_rm, data)  # (N', L, L)
            data.update({'match_mask': match_mask})

            if not self.training:
                anchor_num = match_mask.sum(dim=(1, 2))
                data['non_epipolar'] = anchor_num < self.anchor_num_min

            # perform optimization
            feat_c0, feat_c1 = self.iterative_optimization(feat_c0, feat_c1, match_mask, n_iter, data)  # [N, C, H, W]

            conf_matrix = update_conf_matrix(feat_c0, feat_c1, self.update_weight[n_iter], conf_matrix, data) 

        # 4. coarse matching
        data.update(**get_coarse_match(conf_matrix, self.config['match_coarse'], self.training, data))

        # 5. fine-level pre-process
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)  # [M, C, W, W]

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.threedsam_fine(feat_f0_unfold, feat_f1_unfold)

        # 6. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
