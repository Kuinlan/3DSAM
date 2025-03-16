from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  ThreeDSAM   ↓  ##############
_CN.THREEDSAM = CN()
_CN.THREEDSAM.BACKBONE_TYPE = 'ResNetFPN'
_CN.THREEDSAM.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.THREEDSAM.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.THREEDSAM.FINE_CONCAT_COARSE_FEAT = True

# 0. LoFTR-3dPE config
_CN.THREEDSAM.PE = CN()
_CN.THREEDSAM.PE.DEPTH_MAX = 15.0
# _CN.THREEDSAM.PE.DEPTH_MAX = 1200
_CN.THREEDSAM.PE.DEPTH_MIN = 1e-4
_CN.THREEDSAM.PE.EMBED_DIMS = 256
_CN.THREEDSAM.PE.POSITION_RANGE = [-20.0, -20.0, -8.0, 20.0, 20.0, 20.0]

# 1. loftr backbone & feature fusion config
_CN.THREEDSAM.RESNETFPN = CN()
_CN.THREEDSAM.RESNETFPN.INITIAL_DIM = 128
_CN.THREEDSAM.RESNETFPN.BLOCK_DIMS = [128, 196, 256]

# 2. loftr initialization coarse level attention config
_CN.THREEDSAM.COARSE_INIT = CN()
_CN.THREEDSAM.COARSE_INIT.D_MODEL = 256
_CN.THREEDSAM.COARSE_INIT.D_FFN = 256
_CN.THREEDSAM.COARSE_INIT.NHEAD = 8
_CN.THREEDSAM.COARSE_INIT.LAYER_NAMES = ['self', 'cross'] * 4
_CN.THREEDSAM.COARSE_INIT.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.THREEDSAM.COARSE_INIT.TEMP_BUG_FIX = True

# 3. Coarse-Matching config
_CN.THREEDSAM.MATCH_COARSE = CN()
_CN.THREEDSAM.MATCH_COARSE.MATCH_TYPE = 'dual_softmax' 
_CN.THREEDSAM.MATCH_COARSE.THR = 0.2
_CN.THREEDSAM.MATCH_COARSE.BORDER_RM = 2
_CN.THREEDSAM.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.THREEDSAM.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.THREEDSAM.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.THREEDSAM.MATCH_COARSE.SPARSE_SPVS = True

# 4. fine module config
_CN.THREEDSAM.FINE = CN()
_CN.THREEDSAM.FINE.D_MODEL = 128
_CN.THREEDSAM.FINE.D_FFN = 128
_CN.THREEDSAM.FINE.NHEAD = 8
_CN.THREEDSAM.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.THREEDSAM.FINE.ATTENTION = 'linear'

# 5. Depth Predictor(depth net) config
_CN.THREEDSAM.DEPTH_PREDICTOR = CN()
_CN.THREEDSAM.DEPTH_PREDICTOR.NUM_DEPTH_BINS = 64
# _CN.THREEDSAM.DEPTH_PREDICTOR.NUM_DEPTH_BINS = 128
_CN.THREEDSAM.DEPTH_PREDICTOR.DEPTH_MIN = 1e-4
_CN.THREEDSAM.DEPTH_PREDICTOR.DEPTH_MAX = 15.0
# _CN.THREEDSAM.DEPTH_PREDICTOR.DEPTH_MAX = 1200.0
_CN.THREEDSAM.DEPTH_PREDICTOR.IN_CHANNELS = 256 
_CN.THREEDSAM.DEPTH_PREDICTOR.CONTEXT_CHANNELS = 256 
_CN.THREEDSAM.DEPTH_PREDICTOR.DEPTH_CHANNELS = 64
# _CN.THREEDSAM.DEPTH_PREDICTOR.DEPTH_CHANNELS = 128
_CN.THREEDSAM.DEPTH_PREDICTOR.MID_CHANNELS = 256
_CN.THREEDSAM.DEPTH_PREDICTOR.WITH_CONTEXT_ENCODER = True
_CN.THREEDSAM.DEPTH_PREDICTOR.WITH_DEPTH_CORRECTION = True
_CN.THREEDSAM.DEPTH_PREDICTOR.WITH_PGD = True

# 6. 3DPPE config
_CN.THREEDSAM.PE.DEPTH_MAX = 15
# _CN.THREEDSAM.PE.DEPTH_MAX = 1200
_CN.THREEDSAM.PE.DEPTH_MIN = 1e-4
_CN.THREEDSAM.PE.EMBED_DIMS = 256
_CN.THREEDSAM.PE.POSITION_RANGE = [-20.0, -20.0, -8.0, 20.0, 20.0, 20.0]

# 7. Depth Coarse
_CN.THREEDSAM.DEPTH_COARSE = CN()
_CN.THREEDSAM.DEPTH_COARSE.D_MODEL = 256 # For light-training
_CN.THREEDSAM.DEPTH_COARSE.NHEAD = 8
_CN.THREEDSAM.DEPTH_COARSE.NUM_DEPTH_LAYERS = 2
_CN.THREEDSAM.DEPTH_COARSE.NUM_VISUAL_LAYERS = 2



# 8. LoFTR Losses
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
_CN.THREEDSAM.LOSS.DEPTH_MAX = 15
# _CN.THREEDSAM.LOSS.DEPTH_MAX = 1200
_CN.THREEDSAM.LOSS.DEPTH_MIN = 1e-4
_CN.THREEDSAM.LOSS.NUM_BIN = 128
_CN.THREEDSAM.LOSS.DEPTH_FOCAL_WEIGHT = 0.5
_CN.THREEDSAM.LOSS.DEPTH_DENSE_WEIGHT = 0.5


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
_CN.TRAINER.PLOT_MODE = 'confidence'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'
_CN.TRAINER.PLOT_PATH = 'output/3DPPE_DA_similarity'
_CN.TRAINER.REQUIRE_DEPTH = False

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
