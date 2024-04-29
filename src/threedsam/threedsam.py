import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
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
        self.pos_encoding_8 = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.pos_encoding_16 = PositionEncodingSine(
            config['coarse']['d_model']*2,
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.pos_encoding_32 = PositionEncodingSine(
            config['coarse']['d_model']*4,
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.init_attention = LocalFeatureTransformer(config['coarse'])
        self.iterative_optimization = IterativeOptimization(config)
        self.fine_preprocess = FinePreprocess(config)
        self.threedsam_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

        # for initializing confidence matrix 
        self.iter_num = config['n_iter']
        self.update_weight = config['conf_matrix_update_weight']
        self.temperature = config['match_coarse']['dsmax_temperature']

        # two cases cause skip: 
        # 1. no enough gt matches while training. 
        # 2. no enough matches after init attention while evaluation/testing 
        self.skip = False
        
        # for computing anchor points 
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
        N = data['image0'].shape[0]
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feats_32, feats_16, feats_8 = feats[0:-1]
            feats_f = feats[-1]

            feat0_32, feat1_32 = feats_32.split(data['bs'])
            feat0_16, feat1_16 = feats_16.split(data['bs'])
            feat0_8, feat1_8 = feats_8.split(data['bs'])
            feat0_f, feat1_f = feats_f.split(data['bs'])
            
        else:  # handle different input shapes
            feat0, feat1 = self.backbone(data['image0']), self.backbone(data['image1'])
            feat0_32, feat0_16, feat0_8, feat0_f = feat0
            feat1_32, feat1_16, feat1_8, feat1_f = feat1

        data.update({
            'hw0_c_8': feat0_8.shape[2:], 'hw1_c_8': feat1_8.shape[2:],
            'hw0_c_16': feat0_16.shape[2:], 'hw1_c_16': feat1_16.shape[2:],
            'hw0_c_32': feat0_32.shape[2:], 'hw1_c_32': feat1_32.shape[2:],
            'hw0_f': feat0_f.shape[2:], 'hw1_f': feat1_f.shape[2:]
        })

        # 2. init attention and matching
        feat0_8 = rearrange(self.pos_encoding_8(feat0_8), 'n c h w -> n (h w) c')
        feat1_8 = rearrange(self.pos_encoding_8(feat1_8), 'n c h w -> n (h w) c')

        feat0_16 = rearrange(self.pos_encoding_16(feat0_16), 'n c h w -> n (h w) c')
        feat1_16 = rearrange(self.pos_encoding_16(feat1_16), 'n c h w -> n (h w) c')

        feat0_32 = rearrange(self.pos_encoding_32(feat0_32), 'n c h w -> n (h w) c')
        feat1_32 = rearrange(self.pos_encoding_32(feat1_32), 'n c h w -> n (h w) c')

        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat0_8, feat1_8 = self.init_attention(feat0_8, feat1_8, mask_c0, mask_c1)

        def update_conf_matrix(feat0, feat1, data, weight, key_string='conf_matrix'):
            # normalize
            feat0, feat1 = map(lambda feat: feat / feat.shape[-1]**.5,
                                   [feat0, feat1])
            sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.temperature

            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(), 
                    -INF)
            conf_matrix_new = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2) 

            if key_string in data:
                conf_matrix = weight * conf_matrix_new + (1 - weight) * data[key_string]
            else:
                conf_matrix = conf_matrix_new

            data.update({key_string: conf_matrix})
            
        # initialize confidence matrix
        update_conf_matrix(feat0_8, feat1_8, data, self.update_weight)

        skip_sample = None
        if self.training and data['skip_sample'].sum(0) > 0:  # if no sample need to skip, use all the samples to train 
            skip_sample = data['skip_sample']
            if skip_sample.sum(0) == N:  # no valid sample, whole batch needs to skip
                self.skip = True 
            else:  # split the batch into skip ones and non_skip ones
                feat0_8_skip, feat1_8_skip = feat0_8[skip_sample], feat1_8[skip_sample] 
                conf_matrix_skip = data['conf_matrix'][skip_sample]

                feat0_32, feat1_32 = feat0_32[~skip_sample], feat1_32[~skip_sample]
                feat0_16, feat1_16 = feat0_16[~skip_sample], feat1_16[~skip_sample]
                feat0_8, feat1_8 = feat0_8[~skip_sample], feat1_8[~skip_sample]


        # 3. iterative optimization
        for n in range(self.iter_num):
            match_mask = get_match_mask(data, self.thr, self.border_rm)
            
            # for eval/test mode
            if not self.training:
                anchor_num = match_mask.sum(dim=(1, 2))
                self.skip = True if anchor_num[0] < self.anchor_num_min else False;
            
            # skip iterative optimization
            if self.skip:     
                break
                
            data.update({'match_mask': match_mask})

            # perform iterative optimization
            feats_8, feats_16, feats_32 = IterativeOptimization((feat0_8, feat1_8),
                                                                (feat0_16, feat1_16),
                                                                (feat0_32, feat1_32), data)
            # feature update for next iteration
            feat0_8, feat1_8 = feats_8
            feat0_16, feat1_16 = feats_16
            feat0_32, feat1_32 = feats_32

            update_conf_matrix(feat0_8, feat1_8, data, self.update_weight) 

        # 4. coarse matching 
        if skip_sample is not None and self.skip == False:  # combine two parts splited 
            _, L, S = data['conf_matrix'].shape 

            feat0_8_all = torch.empty((N, L))
            feat1_8_all = torch.empty((N, S))
            feat0_8_all[skip_sample], feat1_8_all[skip_sample] = feat0_8_skip, feat1_8_skip
            feat0_8_all[~skip_sample], feat1_8_all[~skip_sample] = feat0_8, feat1_8
            
            conf_matrix_all = torch.empty((N, L, S))
            conf_matrix_all[skip_sample] = conf_matrix_skip 
            conf_matrix_all[~skip_sample] = data['conf_matrix'] 

            data.update({'conf_matrix': conf_matrix_all})
        
        data.update(**get_coarse_match(data, self.config['match_coarse'], self.training))

        # 5. fine-level pre-process 
        if skip_sample is not None and self.skip == False:
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat0_f, feat1_f, feat0_8_all, feat1_8_all, data)
        else:
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat0_f, feat1_f, feat0_8, feat1_8, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.threedsam_fine(feat_f0_unfold, feat_f1_unfold)
    
        # 6. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)