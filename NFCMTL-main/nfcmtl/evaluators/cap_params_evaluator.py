import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch

from detectron2.data.catalog import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from nfcmtl.utils.utils import save_data_to_csv

logger = logging.getLogger(__name__)


class CapParamsEvaluator(DatasetEvaluator):
    """
    Evaluates MAE of specific keypoint distances for nailfold capillary project.
    Assumes keypoints are [kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8]
    Apex Length: distance(kp1, kp2)
    Venous Length: distance(kp3, kp4)
    Arterial Length: distance(kp5, kp6)
    """

    def __init__(self, dataset_name, output_dir=None):
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        self._pred_apex_lengths = []
        self._pred_venous_lengths = []
        self._pred_arterial_lengths = []

        self._gt_apex_lengths = []
        self._gt_venous_lengths = []
        self._gt_arterial_lengths = []

    def reset(self):
        self._pred_apex_lengths = []
        self._pred_venous_lengths = []
        self._pred_arterial_lengths = []
        self._gt_apex_lengths = []
        self._gt_venous_lengths = []
        self._gt_arterial_lengths = []

    def _calculate_distance(self, kp1, kp2):
        """Calculates Euclidean distance between two keypoints (x, y, ...)."""
        # Ensure keypoints are numpy arrays for easier slicing if they are tensors
        if isinstance(kp1, torch.Tensor):
            kp1 = kp1.cpu().numpy()
        if isinstance(kp2, torch.Tensor):
            kp2 = kp2.cpu().numpy()
        return np.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)

    def process(self, inputs, outputs):
        for input_item, output_item in zip(inputs, outputs):
            # Process predicted keypoints
            if "instances" in output_item:
                pred_instances = output_item["instances"].to(self._cpu_device)
                if pred_instances.has("pred_keypoints"):
                    pred_kpts_all_instances = pred_instances.pred_keypoints # Tensor of shape [num_instances, num_keypoints, 3]
                    pred_classes = pred_instances.pred_classes # Tensor of shape [num_instances]
                    
                    for index, pred_kpts_single_instance in enumerate(pred_kpts_all_instances):
                        if len(pred_kpts_single_instance) >= 6 and pred_classes[index] != 2:  # Do not count hemo class
                            # Apex Length (kp1 and kp2 -> indices 0 and 1)
                            self._pred_apex_lengths.append(self._calculate_distance(pred_kpts_single_instance[0], pred_kpts_single_instance[1]))
                            # Venous Length (kp3 and kp4 -> indices 2 and 3)
                            self._pred_venous_lengths.append(self._calculate_distance(pred_kpts_single_instance[2], pred_kpts_single_instance[3]))
                            # Arterial Length (kp5 and kp6 -> indices 4 and 5)
                            self._pred_arterial_lengths.append(self._calculate_distance(pred_kpts_single_instance[4], pred_kpts_single_instance[5]))
                        else:
                            logger.debug(f"Instance predicted class label {pred_classes[index]}. Skipping distance calculation for this instance.")


            # Process ground truth keypoints
            all_data = DatasetCatalog.get(self._dataset_name)
            for data in all_data:
                if data["file_name"] == input_item["file_name"]:
                    input_item["annotations"] = data.get("annotations", [])
                    break

            if "annotations" in input_item:
                gt_instances = input_item["annotations"]
                for gt_instance in gt_instances:
                        gt_kpts_single_instance = gt_instance.get("keypoints", None)
                        if len(gt_kpts_single_instance) >= 6 * 3 and gt_kpts_single_instance[2] != 0:
                            gt_kpts_tensor = torch.tensor(gt_kpts_single_instance).reshape(-1, 3)


                            self._gt_apex_lengths.append(self._calculate_distance(gt_kpts_tensor[0], gt_kpts_tensor[1]))
                            self._gt_venous_lengths.append(self._calculate_distance(gt_kpts_tensor[2], gt_kpts_tensor[3]))
                            self._gt_arterial_lengths.append(self._calculate_distance(gt_kpts_tensor[4], gt_kpts_tensor[5]))
            else:
                logger.warning("No 'instances' found in input_item. Cannot process ground truth keypoints for MAE calculation.")
        

    def evaluate(self):
        # If using `comm.all_gather`, each process gets all data:
        if comm.get_world_size() > 1:
            # Gather all prediction lists
            comm.synchronize() # Wait for all processes to finish 'process'
            self._pred_apex_lengths = comm.all_gather(self._pred_apex_lengths)
            self._pred_venous_lengths = comm.all_gather(self._pred_venous_lengths)
            self._pred_arterial_lengths = comm.all_gather(self._pred_arterial_lengths)
            
            # Gather all ground truth lists
            self._gt_apex_lengths = comm.all_gather(self._gt_apex_lengths)
            self._gt_venous_lengths = comm.all_gather(self._gt_venous_lengths)
            self._gt_arterial_lengths = comm.all_gather(self._gt_arterial_lengths)

            if not comm.is_main_process():
                return {} # Other processes don't need to compute, just main one

            # Flatten the lists of lists gathered from all processes
            self._pred_apex_lengths = [item for sublist in self._pred_apex_lengths for item in sublist]
            self._pred_venous_lengths = [item for sublist in self._pred_venous_lengths for item in sublist]
            self._pred_arterial_lengths = [item for sublist in self._pred_arterial_lengths for item in sublist]

            self._gt_apex_lengths = [item for sublist in self._gt_apex_lengths for item in sublist]
            self._gt_venous_lengths = [item for sublist in self._gt_venous_lengths for item in sublist]
            self._gt_arterial_lengths = [item for sublist in self._gt_arterial_lengths for item in sublist]


        results = OrderedDict()
        results["cap_params"] = {}

        # --- Apex Length ---
        avg_pred_apex = np.mean(self._pred_apex_lengths) if self._pred_apex_lengths else float('nan')
        avg_gt_apex = np.mean(self._gt_apex_lengths) if self._gt_apex_lengths else float('nan')
        mae_apex_avg = np.abs(avg_pred_apex - avg_gt_apex) if not (np.isnan(avg_pred_apex) or np.isnan(avg_gt_apex)) else float('nan')
        
        results["cap_params"]["avg_pred_apex_length"] = avg_pred_apex
        results["cap_params"]["avg_gt_apex_length"] = avg_gt_apex
        results["cap_params"]["mae_of_avg_apex_length"] = mae_apex_avg


        # --- Venous Length ---
        avg_pred_venous = np.mean(self._pred_venous_lengths) if self._pred_venous_lengths else float('nan')
        avg_gt_venous = np.mean(self._gt_venous_lengths) if self._gt_venous_lengths else float('nan')
        mae_venous_avg = np.abs(avg_pred_venous - avg_gt_venous) if not (np.isnan(avg_pred_venous) or np.isnan(avg_gt_venous)) else float('nan')

        results["cap_params"]["avg_pred_venous_length"] = avg_pred_venous
        results["cap_params"]["avg_gt_venous_length"] = avg_gt_venous
        results["cap_params"]["mae_of_avg_venous_length"] = mae_venous_avg

        # --- Arterial Length ---
        avg_pred_arterial = np.mean(self._pred_arterial_lengths) if self._pred_arterial_lengths else float('nan')
        avg_gt_arterial = np.mean(self._gt_arterial_lengths) if self._gt_arterial_lengths else float('nan')
        mae_arterial_avg = np.abs(avg_pred_arterial - avg_gt_arterial) if not (np.isnan(avg_pred_arterial) or np.isnan(avg_gt_arterial)) else float('nan')

        results["cap_params"]["avg_pred_arterial_length"] = avg_pred_arterial
        results["cap_params"]["avg_gt_arterial_length"] = avg_gt_arterial
        results["cap_params"]["mae_of_avg_arterial_length"] = mae_arterial_avg

        if not results["cap_params"]: # Should not happen with the above structure
             logger.warning("No cap_params results generated.")
             return {}

        logger.info(
            f"NFCMTL Capillary Parameters Evaluator: \n"
            "\n"
            f"Evaluation results for {self._dataset_name} (Capillary Parameters):"
        )
        for key, value in results["cap_params"].items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


        save_data_to_csv(results["cap_params"], os.path.join(self._output_dir, "cap_params_results.csv"))
        
        return results


