import logging
import os
import random

import cv2
import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
from nfcmtl.utils.utils import prepare_torch26


logger = logging.getLogger(__name__)


def visualize(cfg, data_path, output_dir):
    prepare_torch26()

    model = instantiate(cfg.model)
    model.cuda()
    DetectionCheckpointer(model).load(os.path.join(output_dir, "model_final.pth"))

    VAL_IMAGES_PATH = os.path.expanduser(f"{data_path}/val/")
    val_images = os.listdir(VAL_IMAGES_PATH)
    random_image_name = random.choice(val_images)
    print(random_image_name)
    test_image_path = os.path.join(VAL_IMAGES_PATH, random_image_name)

    img = torch.from_numpy(np.ascontiguousarray(read_image(test_image_path, format="BGR")))
    img = img.permute(2, 0, 1)  # HWC -> CHW
    if torch.cuda.is_available():
        img = img.cuda()
    inputs = [{"image": img}]

    model.eval()
    with torch.no_grad():
        predictions_ls = model(inputs)
    predictions = predictions_ls[0]

    # instances = predictions["instances"].to("cpu")

    img_np = img.cpu().numpy().transpose(1, 2, 0)

    # classes = instances.pred_classes.numpy() if hasattr(instances, "pred_classes") else []
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    # class_names = metadata.thing_classes if hasattr(metadata, "thing_classes") else ["unknown"] * len(classes)
    # print(class_names)

    v = Visualizer(img_np, metadata=metadata)

    v = v.draw_instance_predictions(predictions["instances"].to("cpu"))

    predicted_image = v.get_image()[:, :, ::-1]
    # stacked_image = np.vstack((img.cpu(), predicted_image))
    cv2.imwrite(f"./inference/vit_output_image_{random_image_name}.jpg", predicted_image)  # Save the image to a file
    logger.info(f"Successfully saved inference/vit_output_image_{random_image_name}.jpg")
