from functools import partial

import torch
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.data import transforms as T
from detectron2.data.detection_utils import create_keypoint_hflip_indices
from detectron2.evaluation.evaluator import DatasetEvaluators
from detectron2.layers import ShapeSpec
from detectron2.modeling import MViT
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import FPN, BasicStem, ResNet
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch import GeneralizedRCNN
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    CascadeROIHeads,
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    KRCNNConvDeconvUpsampleHead,
    MaskRCNNConvUpsampleHead,
    StandardROIHeads,
)
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from nfcmtl.evaluators.cap_params_evaluator import CapParamsEvaluator
from nfcmtl.evaluators.classification_evaluator import ClassificationEvaluator
from nfcmtl.evaluators.segmentation_evaluator import SegmentationEvaluator

train_set_name = "nailfoldpilot_dataset_train"
val_set_name = "nailfoldpilot_dataset_val"


# Data using LSJ
# dataloader config
image_size = 1024
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
# dataloader = model_zoo.get_config("common/data/coco_keypoint.py").dataloader

dataloader.train.dataset.min_keypoints = 1
dataloader.train.dataset.names = train_set_name
dataloader.test.dataset.names = val_set_name

dataloader.train.mapper.update(
    use_keypoint=True,
    keypoint_hflip_indices=create_keypoint_hflip_indices(dataloader.train.dataset.names),
)


dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 4  # default 64
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]


# dataloader.evaluator = L(COCOEvaluator)(
#     dataset_name="${..test.dataset.names}",
# )

dataloader.evaluator = L(DatasetEvaluators)(
    evaluators=[
        L(ClassificationEvaluator)(dataset_name=val_set_name),
        L(SegmentationEvaluator)(
            dataset_name=val_set_name,
            distributed=True,
            output_dir="segm_test_results",
            num_classes=5
        ),
        L(CapParamsEvaluator)(dataset_name=val_set_name, output_dir="results/keypoints_test_results"),
    ]
)

# dataloader.evaluator.kpt_oks_sigmas = [0.1] * 8
# dataloader.evaluator.max_dets_per_image = 20

# model config
# model = model_zoo.get_config("common/models/keypoint_rcnn_fpn.py").model
model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
model.roi_heads.update(
    num_classes=5,
    keypoint_in_features=["p2", "p3", "p4", "p5"],
    keypoint_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_keypoints=8,
        conv_dims=[512] * 8,
        loss_normalizer="visible",
    ),
)

# Detectron1 uses 2000 proposals per-batch, but this option is per-image in detectron2.
# 1000 proposals per-image is found to hurt box AP.
# Therefore we increase it to 1500 per-image.
model.proposal_generator.post_nms_topk = (1500, 1000)

# Keypoint AP degrades (though box AP improves) when using plain L1 loss
model.roi_heads.box_predictor.smooth_l1_beta = 0.5

constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(
    embed_dim=96,
    depth=10,
    num_heads=1,
    last_block_indexes=(0, 2, 7, 9),
    residual_pooling=True,
    drop_path_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=("scale2", "scale3", "scale4", "scale5"),
)
model.backbone.in_features = "${.bottom_up.out_features}"


uncertainty_loss_settings = {
    "weighted_keys": [
        "loss_rpn_cls",       # RPN classification loss
        "loss_rpn_loc",       # RPN localization loss
        "loss_cls",           # RoI head classification loss
        "loss_box_reg",       # RoI head box regression loss
        "loss_keypoint",      # Keypoint head loss
        "loss_mask",          # Mask head loss
    ],
    "uncertainty_lr_multiplier": 1.0 
}


# trainer config
# max_iter = 67500
max_iter = 10000
train = dict(
    output_dir="./output",
    init_checkpoint="detectron2://ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth",
    max_iter=max_iter,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=True,
    ),
    checkpointer=dict(period=500, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=500,
    log_period=20,
    device="cuda",
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[7500, 9500, max_iter],
    ),
    warmup_length=250 / max_iter,
    warmup_factor=0.001,
)

# optimizer config
optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1.6e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)
optimizer.params.overrides = {
"pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
