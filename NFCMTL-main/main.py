import argparse
import logging
import os
import warnings

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.detection_utils import read_image
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_writers,
    hooks,
)
from detectron2.evaluation import (
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from nfcmtl.evaluators.cap_params_evaluator import CapParamsEvaluator
from nfcmtl.evaluators.classification_evaluator import ClassificationEvaluator
from nfcmtl.evaluators.segmentation_evaluator import SegmentationEvaluator
from nfcmtl.trainer.mtl_trainer import UncertaintyTrainer, UncertaintyTrainerSimp
from nfcmtl.utils.utils import prepare_torch26
from nfcmtl.visualization.visualizer import visualize

# torch.cuda.empty_cache()

keypoint_names = ["up", "down", "left-left", "left-right", "right-left", "right-right", "left-bottom", "right-bottom"]
keypoint_flip_map = [
    ("up", "up"), ("down", "down"),
    ("left-left", "right-right"),
    ("left-right", "right-left"),
    ("left-bottom", "right-bottom")
]

train_set_name = "nailfoldpilot_dataset_train"
val_set_name = "nailfoldpilot_dataset_val"


def register_nailfold_dataset(data_path: str):
    keypoint_attrs = {"keypoint_names": keypoint_names, "keypoint_flip_map": keypoint_flip_map}
    register_coco_instances(
        train_set_name,
        keypoint_attrs,
        os.path.expanduser(data_path + "/train/anfc_coco_train.json"),
        os.path.expanduser(data_path + "/train/")
    )
    register_coco_instances(
        val_set_name,
        keypoint_attrs,
        os.path.expanduser(data_path + "/val/anfc_coco_val.json"),
        os.path.expanduser(data_path + "/val/")
    )


def do_validate(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_test(cfg, model, mode="all"):
    if mode == "all":
        evaluator = cfg.dataloader.evaluator
    elif mode == "classification":
        evaluator = ClassificationEvaluator(cfg.dataloader.test.dataset.names)
    elif mode == "segmentation":
        evaluator = SegmentationEvaluator(
            dataset_name=cfg.dataloader.test.dataset.names,
            distributed=True,
            output_dir="segm_test_results",
            num_classes=5
        )
        # evaluator = SemSegEvaluator(cfg.dataloader.test.dataset.names, distributed=True, output_dir="segm_test_results")
    elif mode == "keypoints":
        evaluator = CapParamsEvaluator(
            dataset_name=cfg.dataloader.test.dataset.names,
            output_dir="results/keypoints_test_results",
        )
    
    ret = inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(evaluator),
    )
    print_csv_format(ret)

    return ret


def setup_cfg(output_dir, exp_name="nfcmtl", save_cfg=False):
    # cfg = LazyConfig.load("detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_h_75ep.py")  # an omegaconf dictionary

    # cfg = LazyConfig.load("detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py")
    cfg = LazyConfig.load("nfcmtl/vit/vit-base-cfg.py")
    # cfg.dataloader.train.dataset.names = TRAIN_SET_NAME
    # cfg.dataloader.test.dataset.names = VAL_SET_NAME
    # cfg.dataloader.train.total_batch_size = 2  # default 64
    # cfg.dataloader.train.num_workers = 0
    # cfg.dataloader.test.num_workers = 0

    cfg.model.roi_heads.batch_size_per_image = 256 # default 512
    cfg.model.proposal_generator.batch_size_per_image = 128 # default 256

    # cfg.model.roi_heads.num_classes = 5
    # cfg.model.roi_heads.keypoint_head.num_keypoints = 8
    # cfg.model.keypoint_on = True


    # cfg.train.checkpointer.max_to_keep = 10  # default 100
    # cfg.train.checkpointer.period = 500  # default 5000

    cfg.train.output_dir = output_dir

    # cfg.test = {"keypoint_oks_sigmas": [0.1] * 8}

    print(cfg)

    if save_cfg:
        LazyConfig.save(cfg, f"vit-config-{exp_name}.yaml")

    return cfg


def do_train(cfg):
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    # trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    # trainer = UncertaintyTrainer(cfg)
    trainer = UncertaintyTrainerSimp(model, train_loader, optim, cfg)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_validate(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)

    trainer.train(0, cfg.train.max_iter)


def nfcmtl_test(cfg, mode: str):
    prepare_torch26()

    model = instantiate(cfg.model)
    model.cuda()
    DetectionCheckpointer(model).load(os.path.join(output_dir, "model_final.pth"))

    do_test(cfg, model, mode)


def setup_logging(exp_name="nfcmtl"):
    # logging
    log_filename = f"logs/training_log_vit_{exp_name}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NFCMTL.')
    parser.add_argument('--data-path', type=str, default=None, help='Path to the dataset folder.')

    data_path = parser.parse_args().data_path

    training_seq = 31
    training_focus = "uncertainty-10k-iter"
    exp_name = f"{training_seq}-{training_focus}"
    output_dir = f"output/vit/{exp_name}"
    register_nailfold_dataset(data_path)
    cfg = setup_cfg(output_dir, exp_name, save_cfg=False)
    setup_logging(exp_name)

    # do_train(cfg)
    
    # nfcmtl_test(cfg, mode="segmentation")
    # nfcmtl_test(cfg, mode="classification")
    nfcmtl_test(cfg, mode="keypoints")

    # visualize(cfg, data_path, output_dir)
