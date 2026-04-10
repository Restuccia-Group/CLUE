from yacs.config import CfgNode

_C = CfgNode()
cfg = _C
# ------------ Base Options ---------------------------#
_C.BASE = CfgNode()
_C.BASE.SEED = 10
_C.BASE.NUM_WORKERS = 8
_C.BASE.GPU_ID = 0
# --------------- DATASET Options ---------------------#
_C.DATA = CfgNode()
_C.DATASET = "cifar10"
_C.DATA.MODE = 'test'
#_C.DATA.AUG_SET = ['normal','brightness', 'contrast','saturation', 'hue', 'gamma','gaussian_blur', 'gaussian_noise']
_C.DATA.AUG = 'normal'
_C.DATA.SEVERITY = 5
_C.DATA.CORRUPTION = 'normal'
_C.DATA.SEVERITY = 5
_C.DATA.CORRUPTION_SET = ['normal','gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                          'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 
                          'pixelate', 'jpeg_compression']

_C.DATA.CORRUPTION_SET_UNSEEN = ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise']

# ------------------- Train Options --------------#
_C.TR = CfgNode()
_C.TR.MODE = 'testing'
_C.TR.N_EPOCH = 20
_C.TR.BATCH_SIZE = 64

# -------------------- Test Options --------------#
_C.TST = CfgNode()
_C.TST.BATCH_SIZE = 64

# ----- Optimizer Options -------------------------#
_C.OPTIM = CfgNode()
_C.OPTIM.STEPS = 1 
_C.OPTIM.LR = 0.001
_C.OPTIM.METHOD = 'Adam'

# ----- Path Options ------------#
_C.PATH = CfgNode()
_C.PATH.DATA_PATH = 'corrupted_data/severity_5'
_C.PATH.SOURCE_DATA = 'data'
_C.PATH.NOISE_ENC = 'saved_model/noise_encoder/cifar10c.pt'
_C.PATH.MODEL_NAME = 'resenet50ch'
_C.PATH.SAVED_MODEL= 'saved_model_aug' + '/' + _C.PATH.MODEL_NAME 
_C.PATH.SOURCE_MODEL = 'saved_model/saved_model_data_aug.pt'

# ----- noise encoder options ------
_C.NOISE_ENC = CfgNode()
_C.NOISE_ENC.TEMP = 0.1
_C.NOISE_ENC.EMBED_DIM = 128
_C.NOISE_ENC.LAMBDA_EMBED = 0.005
_C.NOISE_ENC.LR = 0.001
_C.NOISE_ENC.N_EPOCH = 200
_C.NOISE_ENC.BATCH_SIZE = 128

