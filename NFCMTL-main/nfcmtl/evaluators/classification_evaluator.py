import logging
import torch
from collections import OrderedDict

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

logger = logging.getLogger(__name__)

class ClassificationEvaluator(DatasetEvaluator):
    """
    Evaluator for a classification task, focusing on whether images containing
    an "abnormal" capillary (class ID 3) are correctly identified as such.
    A "hit" occurs if an image with a ground truth "abnormal" capillary also has
    a predicted "abnormal" capillary.
    The primary metric is the ratio of these hits to the total number of images
    with ground truth "abnormal" capillaries, reported as "abnormal_detection_accuracy".
    Ground truth annotations are retrieved from DatasetCatalog by matching file_name.
    """

    def __init__(self, dataset_name, distributed=True, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        
        self._gt_abnormal_image_count = 0  # Number of images with at least one GT abnormal instance
        self._successful_hit_count = 0     # Number of hits (GT abnormal image, and Pred abnormal image)

    def reset(self):
        """
        Resets the internal state for a new evaluation.
        """
        self._gt_abnormal_image_count = 0
        self._successful_hit_count = 0

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        Args:
            inputs (list): the inputs that's used to call the model. Each item
                           is a dict containing at least "file_name".
            outputs (list): the return value of the model, one for each input.
        """
        for input_per_image, output_per_image in zip(inputs, outputs):
            # --- Prediction Processing ---
            if "instances" not in output_per_image:
                logger.warning(f"No 'instances' found in output for {input_per_image.get('file_name', 'unknown image')}. Skipping.")
                continue
            
            pred_instances = output_per_image["instances"].to(self._cpu_device)
            if not hasattr(pred_instances, 'pred_classes'):
                logger.warning(f"No 'pred_classes' found in predicted instances for {input_per_image.get('file_name', 'unknown image')}. Skipping.")
                continue
            
            pred_classes_set = set(pred_instances.pred_classes.tolist())
            image_predicted_as_abnormal = (3 in pred_classes_set)

            # --- Ground Truth Retrieval ---
            input_file_name = input_per_image.get("file_name")
            if not input_file_name:
                logger.warning("Input item missing 'file_name'. Cannot retrieve ground truth. Skipping.")
                continue

            gt_annotations = None
            dataset_dicts = DatasetCatalog.get(self._dataset_name)
            for data_item in dataset_dicts:
                if data_item["file_name"] == input_file_name:
                    if "annotations" in data_item and data_item["annotations"]:
                        gt_annotations = data_item["annotations"]
                    break
            
            if gt_annotations is None:
                logger.warning(
                    f"No ground truth annotations found in DatasetCatalog for {input_file_name}. "
                    "This image will be skipped for evaluation."
                )
                continue
            
            gt_classes_set = set(anno["category_id"] for anno in gt_annotations)
            image_is_gt_abnormal = (3 in gt_classes_set)

            # --- Update Counters Based on Original Logic ---
            if image_is_gt_abnormal:
                self._gt_abnormal_image_count += 1
                if image_predicted_as_abnormal:
                    self._successful_hit_count += 1

    def evaluate(self):
        """
        Evaluates the accumulated "hits".
        Returns:
            dict: A dictionary of metrics.
        """
        if self._distributed:
            comm.synchronize()
            # Gather counters
            gt_abnormal_image_counts_gathered = comm.gather(self._gt_abnormal_image_count, dst=0)
            successful_hit_counts_gathered = comm.gather(self._successful_hit_count, dst=0)

            if not comm.is_main_process():
                return {}
            
            total_gt_abnormal_images = sum(gt_abnormal_image_counts_gathered)
            total_successful_hits = sum(successful_hit_counts_gathered)
        else:
            total_gt_abnormal_images = self._gt_abnormal_image_count
            total_successful_hits = self._successful_hit_count

        if total_gt_abnormal_images == 0:
            logger.warning(f"No ground truth images with abnormal capillaries (class ID 3) found in dataset '{self._dataset_name}'. Accuracy cannot be calculated.")
            return {"abnormal_detection_accuracy": float('nan'), "total_gt_abnormal_images": 0, "total_successful_hits": 0}

        accuracy_value = total_successful_hits / total_gt_abnormal_images
        
        results = OrderedDict()
        results["abnormal_detection_accuracy"] = accuracy_value
        results["total_gt_abnormal_images"] = total_gt_abnormal_images
        results["total_successful_hits"] = total_successful_hits
        
        logger.info(
            "NFCMTL Classification Evaluator: \n"
            "\n"
            f"Classification results for dataset '{self._dataset_name}': "
            f"Abnormal Detection Accuracy: {accuracy_value:.4f} "
            f"(Hits: {total_successful_hits} / Total GT Abnormal Images: {total_gt_abnormal_images})"
        )
        
        return results
