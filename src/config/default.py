from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  DPT   ↓  ##############
_CN.DPT = CN()
_CN.DPT.WEIGHT_PATH = '/home/morgen/projects/3DSAM/weights/dpt/dpt_hybrid_nyu-2ce69ec7.pt'
_CN.DPT.OPTIMIZE = True

##############  ↓  ThreeDSAM   ↓  ##############
_CN.THREEDSAM = CN()
_CN.THREEDSAM.N_ITER = 4
_CN.THREEDSAM.LAYER_ASSIGN = [0, 1, 2, 3]
_CN.THREEDSAM.BACKBONE_TYPE = 'ResNetFPN'
_CN.THREEDSAM.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.THREEDSAM.D_STRUCT = 128
_CN.THREEDSAM.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.THREEDSAM.FINE_CONCAT_COARSE_FEAT = True

# 1. 3DSAM-backbone & feature fusion config
_CN.THREEDSAM.RESNETFPN = CN()
_CN.THREEDSAM.RESNETFPN.INITIAL_DIM = 64
_CN.THREEDSAM.RESNETFPN.BLOCK_DIMS = [64, 128, 256]

# 2. 3DSAM Init attention config
_CN.THREEDSAM.COARSE = CN()
_CN.THREEDSAM.COARSE.MP = False
_CN.THREEDSAM.COARSE.HALF = False
_CN.THREEDSAM.COARSE.D_MODEL = 256
_CN.THREEDSAM.COARSE.D_FFN = 256
_CN.THREEDSAM.COARSE.NHEAD = 8
_CN.THREEDSAM.COARSE.LAYER_NAMES = ['self', 'cross']
_CN.THREEDSAM.COARSE.AGG_SIZE0 = 4
_CN.THREEDSAM.COARSE.AGG_SIZE1 = 4
_CN.THREEDSAM.COARSE.NO_FLASH = False
_CN.THREEDSAM.COARSE.ROPE = True  
_CN.THREEDSAM.COARSE.NPE = [480, 640, 480, 640]
_CN.THREEDSAM.COARSE.NPE = [480, 640, 480, 640]

# 3.1. 3DSAM-feature extractor config
_CN.THREEDSAM.EXTRACTOR = CN()
_CN.THREEDSAM.EXTRACTOR.ANCHOR_NUM = 32  # for training
_CN.THREEDSAM.EXTRACTOR.ANCHOR_NUM_MIN = 8  # for test/val
_CN.THREEDSAM.EXTRACTOR.ANCHOR_THR = 0.5 
_CN.THREEDSAM.EXTRACTOR.TRAIN_PAD_ANCHOR_NUM_MIN = 8
_CN.THREEDSAM.EXTRACTOR.ANCHOR_SAMPLER_SEED = 66
_CN.THREEDSAM.EXTRACTOR.BORDER_RM = 2
_CN.THREEDSAM.EXTRACTOR.D_COLOR = 256
_CN.THREEDSAM.EXTRACTOR.D_STRUCT = 128
_CN.THREEDSAM.EXTRACTOR.DROPOUT_PROB = 0.2

# 3.2. epipolar-cross-attention module config
_CN.THREEDSAM.EPIPOLAR_ATTENTION = CN()
_CN.THREEDSAM.EPIPOLAR_ATTENTION.MP = False
_CN.THREEDSAM.EPIPOLAR_ATTENTION.HALF = False
_CN.THREEDSAM.EPIPOLAR_ATTENTION.AREA_WIDTH = 4
_CN.THREEDSAM.EPIPOLAR_ATTENTION.D_MODEL = 256 
_CN.THREEDSAM.EPIPOLAR_ATTENTION.NHEAD = 8 
_CN.THREEDSAM.EPIPOLAR_ATTENTION.LAYER_NAMES = ['cross'] * 1
_CN.THREEDSAM.EPIPOLAR_ATTENTION.AGG_SIZE0 = 4
_CN.THREEDSAM.EPIPOLAR_ATTENTION.AGG_SIZE1 = 4
_CN.THREEDSAM.EPIPOLAR_ATTENTION.NO_FLASH = False
_CN.THREEDSAM.EPIPOLAR_ATTENTION.ROPE = False
_CN.THREEDSAM.EPIPOLAR_ATTENTION.NPE = [480, 640, 480, 640]

# 3.3 self-attention module config
_CN.THREEDSAM.SELF_ATTENTION = CN()
_CN.THREEDSAM.SELF_ATTENTION.MP = False
_CN.THREEDSAM.SELF_ATTENTION.HALF = False
_CN.THREEDSAM.SELF_ATTENTION.AREA_WIDTH = None
_CN.THREEDSAM.SELF_ATTENTION.D_MODEL = 256
_CN.THREEDSAM.SELF_ATTENTION.NHEAD = 8
_CN.THREEDSAM.SELF_ATTENTION.LAYER_NAMES = ['self'] * 1
_CN.THREEDSAM.SELF_ATTENTION.AGG_SIZE0 = 4
_CN.THREEDSAM.SELF_ATTENTION.AGG_SIZE1 = 4
_CN.THREEDSAM.SELF_ATTENTION.NO_FLASH = False
_CN.THREEDSAM.SELF_ATTENTION.ROPE = False,  
_CN.THREEDSAM.SELF_ATTENTION.NPE = [480, 640, 480, 640]

# 4. Coarse-Matching config
_CN.THREEDSAM.MATCH_COARSE = CN()
_CN.THREEDSAM.MATCH_COARSE.MATCH_TYPE = 'dual_softmax' 
_CN.THREEDSAM.MATCH_COARSE.THR = 0.2
_CN.THREEDSAM.MATCH_COARSE.BORDER_RM = 2
_CN.THREEDSAM.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.THREEDSAM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.THREEDSAM.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.THREEDSAM.MATCH_COARSE.SPARSE_SPVS = True

# 5. fine module config
_CN.THREEDSAM.FINE = CN()
_CN.THREEDSAM.FINE.MP = False
_CN.THREEDSAM.FINE.HALF = False
_CN.THREEDSAM.FINE.AREA_WIDTH = None
_CN.THREEDSAM.FINE.D_MODEL = 64
_CN.THREEDSAM.FINE.NHEAD = 8
_CN.THREEDSAM.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.THREEDSAM.FINE.AGG_SIZE0 = 1
_CN.THREEDSAM.FINE.AGG_SIZE1 = 1
_CN.THREEDSAM.FINE.NO_FLASH = False
_CN.THREEDSAM.FINE.ROPE = True  
_CN.THREEDSAM.FINE.NPE = [480, 640, 480, 640]

# 6. LoFTR Losses
# -- # coarse-level
_CN.THREEDSAM.LOSS = CN()
_CN.THREEDSAM.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.THREEDSAM.LOSS.COARSE_WEIGHT = 1.0
# _CN.THREEDSAM.LOSS.SPARSE_SPVS = False
# -- - -- # focal loss (coarse)
_CN.THREEDSAM.LOSS.FOCAL_ALPHA = 0.25
_CN.THREEDSAM.LOSS.FOCAL_GAMMA = 2.0
_CN.THREEDSAM.LOSS.POS_WEIGHT = 1.0
_CN.THREEDSAM.LOSS.NEG_WEIGHT = 1.0
# _CN.THREEDSAM.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# use `_CN.THREEDSAM.MATCH_COARSE.MATCH_TYPE`

# -- # fine-level
_CN.THREEDSAM.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.THREEDSAM.LOSS.FINE_WEIGHT = 1.0
_CN.THREEDSAM.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# ##############  ↓  LoFTR Pipeline  ↓  ##############
# _CN.LOFTR = CN()
# _CN.LOFTR.BACKBONE_TYPE = 'ResNetFPN'
# _CN.LOFTR.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
# _CN.LOFTR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
# _CN.LOFTR.FINE_CONCAT_COARSE_FEAT = True

# # 1. LoFTR-backbone (local feature CNN) config
# _CN.LOFTR.RESNETFPN = CN()
# _CN.LOFTR.RESNETFPN.INITIAL_DIM = 128
# _CN.LOFTR.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# # 2. LoFTR-coarse module config
# _CN.LOFTR.COARSE = CN()
# _CN.LOFTR.COARSE.D_MODEL = 256
# _CN.LOFTR.COARSE.D_FFN = 256
# _CN.LOFTR.COARSE.NHEAD = 8
# _CN.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross']
# _CN.LOFTR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
# _CN.LOFTR.COARSE.TEMP_BUG_FIX = True

# # 3. Coarse-Matching config
# _CN.LOFTR.MATCH_COARSE = CN()
# _CN.LOFTR.MATCH_COARSE.THR = 0.2
# _CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
# _CN.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
# _CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
# _CN.LOFTR.MATCH_COARSE.SKH_ITERS = 3
# _CN.LOFTR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
# _CN.LOFTR.MATCH_COARSE.SKH_PREFILTER = False
# _CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
# _CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
# _CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True

# # 4. LoFTR-fine module config
# _CN.LOFTR.FINE = CN()
# _CN.LOFTR.FINE.D_MODEL = 128
# _CN.LOFTR.FINE.D_FFN = 128
# _CN.LOFTR.FINE.NHEAD = 8
# _CN.LOFTR.FINE.LAYER_NAMES = ['self', 'cross'] * 1
# _CN.LOFTR.FINE.ATTENTION = 'linear'

# # 5. LoFTR Losses
# # -- # coarse-level
# _CN.LOFTR.LOSS = CN()
# _CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
# _CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
# # _CN.LOFTR.LOSS.SPARSE_SPVS = False
# # -- - -- # focal loss (coarse)
# _CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
# _CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
# _CN.LOFTR.LOSS.POS_WEIGHT = 1.0
# _CN.LOFTR.LOSS.NEG_WEIGHT = 1.0
# # _CN.LOFTR.LOSS.DUAL_SOFTMAX = False  # whether coarse-level use dual-softmax or not.
# # use `_CN.LOFTR.MATCH_COARSE.MATCH_TYPE`

# # -- # fine-level
# _CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
# _CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
# _CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
