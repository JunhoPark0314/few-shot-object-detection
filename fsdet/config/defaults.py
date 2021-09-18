from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True

# EMA_M for ema model
_CC.SOLVER.EMA_M = 0.999

_CC.TRAINER = "TFA"

_CC.MEMORY = CN()
_CC.MEMORY.UPDATE_PERIOD = 100
_CC.MEMORY.NUM_IMAGE = 200
_CC.MEMORY.NUM_INSTANCE = 200