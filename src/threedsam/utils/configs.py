from yacs.config import CfgNode as CN
_CN = CN()


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

##############  ↓  ThreeDSAM   ↓  ##############
_CN.N_ITER = 4
_CN.CONF_MATRIX_UPDATE_WEIGHT = 0.5
_CN.BACKBONE_TYPE = 'ResNetFPN'
_CN.RESOLUTION = (32, 16, 8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True

# 1. 3DSAM-backbone & feature fusion config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 64
_CN.RESNETFPN.BLOCK_DIMS = [64, 128, 256, 512, 768]

# 2. 3DSAM Init attention config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross']
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.COARSE.TEMP_BUG_FIX = True

# 3.1. 3DSAM-feature extractor config
_CN.EXTRACTOR = CN()
_CN.EXTRACTOR.ANCHOR_NUM = 32  # for training
_CN.EXTRACTOR.ANCHOR_NUM_MIN = 32  # for test/val
_CN.EXTRACTOR.ANCHOR_THR = 0.5 
_CN.EXTRACTOR.TRAIN_PAD_ANCHOR_NUM_MIN = 8
_CN.EXTRACTOR.ANCHOR_SAMPLER_SEED = 66
_CN.EXTRACTOR.BORDER_RM = 2
_CN.EXTRACTOR.D_COLOR = 256
_CN.EXTRACTOR.D_STRUCT = 128
_CN.EXTRACTOR.DROPOUT_PROB = 0.2

# 3.2. Epipolar-Attention module config
_CN.EPIPOLAR_ATTENTION = CN()
_CN.EPIPOLAR_ATTENTION.AREA_WIDTH = 4
_CN.EPIPOLAR_ATTENTION.D_MODEL = 768
_CN.EPIPOLAR_ATTENTION.D_FFN = 256
_CN.EPIPOLAR_ATTENTION.NHEAD = 16
_CN.EPIPOLAR_ATTENTION.ATTENTION = 'linear'

# 3.3 self-attention module config
_CN.SELF_ATTENTION = CN()
_CN.SELF_ATTENTION.D_MODEL = 512
_CN.SELF_ATTENTION.NHEAD = 8
_CN.SELF_ATTENTION.ATTENTION = 'linear'
_CN.SELF_ATTENTION.LAYER_NAMES = ['self']

# 4. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax' 
_CN.MATCH_COARSE.THR = 0.2
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.MATCH_COARSE.SPARSE_SPVS = True

# 5. fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 64
_CN.FINE.D_FFN = 128
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'

default_cfg = lower_config(_CN)