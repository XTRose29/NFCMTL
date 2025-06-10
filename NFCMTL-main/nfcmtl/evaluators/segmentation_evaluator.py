import logging
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import skimage as ski
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import BitMasks
from detectron2.structures.instances import Instances
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class SegmentationEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None, *, num_classes=None, ignore_label=None):
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        self.meta = meta # Store metadata for use in process

        if num_classes is None:
            try:
                num_classes = len(meta.stuff_classes)
            except AttributeError:
                logger.error(f"Metadata for {dataset_name} must contain 'stuff_classes' attribute.")
                raise ValueError(f"num_classes not provided and 'stuff_classes' not found in metadata for {dataset_name}.")
        self._num_classes = num_classes

        if ignore_label is None:
            try:
                ignore_label = meta.ignore_label
            except AttributeError:
                logger.warning(
                    f"'ignore_label' not provided / not found in metadata for {dataset_name}. "
                    "Defaulting to 255."
                )
                ignore_label = 255 # Common default
        self._ignore_label = ignore_label
        
        self._class_names = meta.stuff_classes if hasattr(meta, "stuff_classes") else [str(i) for i in range(self._num_classes)]
        logger.info(f"Custom evaluator initialized for {self._num_classes} classes: {self._class_names}")
        logger.info(f"Using ignore_label: {self._ignore_label}")
        
        if not hasattr(self.meta, "thing_dataset_id_to_contiguous_id"):
            logger.warning(f"'thing_dataset_id_to_contiguous_id' not found in metadata for {dataset_name}. "
                                 "Ensure class IDs from instances are directly usable or mapping is handled.")

        self._conf_matrix = None
        self.reset()

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)

    def _generate_gt_sem_seg(self, input_data):
        """
        Generates the ground truth semantic segmentation mask from instance annotations.
        Relies on 'instances' (an Instances object) or 'annotations' (raw COCO dicts) in input_data.
        The 'instances' field is typically populated by Detectron2's DatasetMapper.
        """
        if "height" not in input_data or "width" not in input_data:
            logger.error("Input data missing 'height' or 'width'. Cannot generate GT mask.")
            return None
        
        all_data = DatasetCatalog.get(self._dataset_name)
        for data in all_data:
            if data["file_name"] == input_data["file_name"]:
                input_data["annotations"] = data.get("annotations", [])
                break


        h, w = input_data["height"], input_data["width"]
        sem_seg_gt_np = np.full((h, w), self._ignore_label, dtype=np.uint8)

        if "instances" in input_data:
            instances = input_data["instances"] 
            if instances.has("gt_masks") and instances.has("gt_classes"):
                gt_masks_obj = instances.gt_masks 
                gt_classes_tensor = instances.gt_classes.to(self._cpu_device)

                for i in range(len(instances)):
                    class_id = gt_classes_tensor[i].item()
                    
                    if not (0 <= class_id < self._num_classes):
                        # This check is important if thing_classes (from COCO) are used to generate gt_classes
                        # but stuff_classes (for the evaluator) are different.
                        # Assumes gt_classes are already mapped to be valid indices for stuff_classes.
                        # logger.debug(f"GT class_id {class_id} is out of semantic range [0, {self._num_classes-1}]. It might be ignored or indicate a mapping issue.")
                        pass # Let it be painted; filtering against ignore_label and valid range happens later

                    mask_i_tensor = None
                    if isinstance(gt_masks_obj, BitMasks):
                        mask_i_tensor = gt_masks_obj.tensor[i] 
                    # Add robust handling if gt_masks_obj could be PolygonMasks and not yet converted by mapper
                    # For now, assume DatasetMapper provides BitMasks if masks are used.
                    else:
                        logger.warning(f"Unsupported gt_masks type: {type(gt_masks_obj)} in _generate_gt_sem_seg. Expecting BitMasks.")
                        continue
                        
                    mask_i_np = mask_i_tensor.cpu().numpy() # Should be (h, w) boolean
                    sem_seg_gt_np[mask_i_np] = class_id 
            else:
                logger.debug(f"No 'gt_masks' or 'gt_classes' in 'instances' for {input_data.get('file_name', 'Unknown file')}. GT mask will be empty (all ignore_label).")
        
        elif "annotations" in input_data and input_data["annotations"]:
            logger.debug("Processing raw 'annotations' to generate GT semantic mask for {input_data.get('file_name', 'Unknown file')}.")
            if not hasattr(self.meta, "thing_dataset_id_to_contiguous_id"):
                logger.error("'thing_dataset_id_to_contiguous_id' needed in metadata to process raw annotations. Skipping GT generation.")
                return None

            for ann in input_data["annotations"]:
                if ann.get("iscrowd", 0):
                    continue
                if "segmentation" in ann:
                    coco_class_id = ann["category_id"]
                    class_id = self.meta.thing_dataset_id_to_contiguous_id.get(coco_class_id, self._ignore_label)
                    
                    if class_id == self._ignore_label: # Already an ignore label by mapping
                        continue
                    # class_id here is a contiguous "thing" ID. It should align with a "stuff" ID.
                    if not (0 <= class_id < self._num_classes): # Check if this mapped ID is valid for stuff classes
                        # logger.debug(f"Mapped GT class_id {class_id} (from COCO {coco_class_id}) is out of semantic range [0, {self._num_classes-1}].")
                        continue

                    mask_rle = None
                    if isinstance(ann["segmentation"], list): 
                        rles = mask_util.frPyObjects(ann["segmentation"], h, w)
                        mask_rle = mask_util.merge(rles)
                    elif isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]: 
                        mask_rle = ann["segmentation"]
                    else:
                        continue 
                    
                    mask_i_np = mask_util.decode(mask_rle) # Binary mask (h,w)
                    sem_seg_gt_np[mask_i_np > 0] = class_id # Use > 0 for binary mask
        else:
            logger.debug(f"No 'instances' or 'annotations' found in input for {input_data.get('file_name', 'Unknown file')}. GT mask will be empty.")

        return torch.as_tensor(sem_seg_gt_np.astype("long"))


    def process(self, inputs, outputs):
        for input_data, output_data_item in zip(inputs, outputs):
            # 1. Get/Generate predicted semantic segmentation mask (pred_seg)
            pred_seg_np = None
            # Determine image dimensions, preferably from input_data for consistency with GT.
            # Fallback to output_data_item if it has image_size (e.g. Instances object)
            image_h, image_w = input_data["height"], input_data["width"]

            output_data_item = output_data_item["instances"]
            if isinstance(output_data_item, Instances):
                pred_instances = output_data_item
                # logger.debug(f"Processing Instances object for predictions: {pred_instances}")
                
                # Use image_size from pred_instances if it exists and is consistent.
                # For now, we build the mask with image_h, image_w from input_data.
                # pred_masks from Instances are typically full image resolution.
                _pred_sem_seg_buffer_np = np.full((image_h, image_w), self._ignore_label, dtype=np.uint8)

                if pred_instances.has("pred_masks") and pred_instances.has("pred_classes"):
                    # pred_masks is typically a tensor of shape (N, H, W)
                    pred_masks_tensor = pred_instances.pred_masks 
                    # pred_classes is typically a tensor of shape (N,)
                    pred_classes_tensor = pred_instances.pred_classes.to(self._cpu_device) 

                    # Optional: Sort by scores if available, so higher confidence predictions are painted last
                    # This handles overlaps. If scores aren't used, order in Instances object dictates.
                    indices = range(len(pred_instances))
                    if pred_instances.has("scores"):
                        scores = pred_instances.scores.to(self._cpu_device).numpy()
                        indices = np.argsort(scores) # Sort by score, lowest to highest

                    for i in indices: # Iterate (potentially sorted by score)
                        class_id = pred_classes_tensor[i].item()
                        
                        # Ensure predicted class_id is valid for the evaluator's num_classes
                        if not (0 <= class_id < self._num_classes):
                            logger.warning(f"Predicted class_id {class_id} is out of evaluator's range [0, {self._num_classes-1}]. Skipping this predicted instance.")
                            continue 
                        
                        # mask_i_tensor is one instance's mask, e.g., (H_instance, W_instance)
                        # It should be at the full resolution matching image_h, image_w
                        mask_i_tensor = pred_masks_tensor[i] 
                        mask_i_np = mask_i_tensor.cpu().numpy() # Boolean (image_h, image_w)
                        
                        _pred_sem_seg_buffer_np[mask_i_np] = class_id # Paint the class
                    
                    pred_seg_np = _pred_sem_seg_buffer_np
                else:
                    logger.warning(f"Output Instances object for {input_data.get('file_name', 'Unknown file')} "
                                         "is missing 'pred_masks' or 'pred_classes'. Cannot generate predicted semantic mask from instances.")
            
            elif isinstance(output_data_item, dict) and "sem_seg" in output_data_item:
                # This path is for models that directly output 'sem_seg' logits (e.g., pure semantic segmentation models)
                logger.debug(f"Processing dict with 'sem_seg' key for predictions for {input_data.get('file_name', 'Unknown file')}")
                pred_seg_logits = output_data_item["sem_seg"] # Expected (C, H, W)
                pred_seg_np = pred_seg_logits.argmax(dim=0).to(self._cpu_device).numpy()
            
            if pred_seg_np is None:
                logger.warning(f"Could not derive predicted semantic mask from output for {input_data.get('file_name', 'Unknown file')}. Skipping item.")
                continue

            # 2. Generate ground truth semantic segmentation mask (gt_seg)
            gt_seg_tensor = self._generate_gt_sem_seg(input_data)
            if gt_seg_tensor is None:
                logger.warning(f"Could not generate GT semantic mask for {input_data.get('file_name', 'Unknown file')}. Skipping.")
                continue
            gt_seg_np = gt_seg_tensor.cpu().numpy() # Already (H,W) with class labels

            # 3. Continue with evaluation logic (comparing pred_seg_np and gt_seg_np)
            if pred_seg_np.shape != gt_seg_np.shape:
                logger.warning(
                    f"Shape mismatch for item {input_data.get('file_name', '')}. "
                    f"GT shape: {gt_seg_np.shape}, Pred shape: {pred_seg_np.shape}. Resizing pred to GT."
                )
                try:
                    # Ensure pred_seg_np is uint8 if it's not already, for resize order=0
                    pred_seg_np = ski.resize(pred_seg_np.astype(np.uint8), gt_seg_np.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                except ImportError:
                    logger.error("scikit-image not found. Cannot resize. Please install it: pip install scikit-image")
                    continue # Skip this item if resize fails

            gt_flat = gt_seg_np.flatten()
            pred_flat = pred_seg_np.flatten()

            # --- Crucial Filtering Logic ---
            # 1. Mask for pixels where GT is not an ignore label
            gt_not_ignore_mask = (gt_flat != self._ignore_label)
            
            # 2. Mask for pixels where GT is a valid class index (within [0, num_classes-1])
            # This is important if GT might contain labels outside this range for some reason,
            # even after ignore_label filtering.
            gt_valid_class_idx_mask = (gt_flat >= 0) & (gt_flat < self._num_classes)
            
            # 3. Mask for pixels where prediction is a valid class index (within [0, num_classes-1])
            # This is critical. Predictions might be ignore_label if no instance covers a pixel,
            # or if a predicted class_id was filtered out earlier.
            pred_valid_class_idx_mask = (pred_flat >= 0) & (pred_flat < self._num_classes)

            # Combine all masks:
            # - GT must not be an ignore_label.
            # - GT must be a valid class index for the confusion matrix.
            # - Prediction must also be a valid class index for the confusion matrix.
            final_valid_mask = gt_not_ignore_mask & gt_valid_class_idx_mask & pred_valid_class_idx_mask
            
            gt_flat_filtered = gt_flat[final_valid_mask]
            pred_flat_filtered = pred_flat[final_valid_mask]

            if gt_flat_filtered.size == 0: # No valid pixels to evaluate after all filtering
                logger.debug(f"No valid pixels remaining after filtering for {input_data.get('file_name', 'Unknown file')}.")
                continue

            # Both gt_flat_filtered and pred_flat_filtered now only contain values in [0, self._num_classes - 1]
            current_conf_matrix = np.bincount(
                self._num_classes * gt_flat_filtered.astype(np.int64) + pred_flat_filtered.astype(np.int64),
                minlength=self._num_classes**2
            ).reshape(self._num_classes, self._num_classes)
            
            self._conf_matrix += current_conf_matrix

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            conf_matrix_list = comm.gather(self._conf_matrix, dst=0)
            if comm.is_main_process():
                self._conf_matrix = np.zeros_like(self._conf_matrix)
                for conf_m in conf_matrix_list:
                    self._conf_matrix += conf_m
            else:
                return {} 
        
        if not comm.is_main_process() and self._distributed:
             return {}

        if self._conf_matrix.sum() == 0:
            logger.warning("Confusion matrix is all zeros. No samples were processed or all samples were ignored.")
            results = OrderedDict()
            results["mIoU"] = 0.0
            results["pixel_accuracy"] = 0.0
            results["macro_sensitivity"] = 0.0 # Add macro_sensitivity
            for i, name in enumerate(self._class_names):
                results[f"IoU-{name}"] = 0.0
                results[f"sensitivity-{name}"] = 0.0 # Changed from accuracy to sensitivity
            return results
        
        # True Positives (TP): Diagonal elements
        tp = np.diag(self._conf_matrix)
        # False Positives (FP): Sum of columns minus TP
        fp = self._conf_matrix.sum(axis=0) - tp
        # False Negatives (FN): Sum of rows minus TP
        fn = self._conf_matrix.sum(axis=1) - tp

        epsilon = 1e-7 
        
        # Intersection over Union (IoU) per class
        iou_per_class = tp / (tp + fp + fn + epsilon) 
        mean_iou = np.nanmean(iou_per_class) # Ignore NaN for classes not in GT/Pred

        # Sensitivity (Recall or True Positive Rate) per class
        # Sensitivity = TP / (TP + FN)
        sensitivity_per_class = tp / (tp + fn + epsilon)
        macro_sensitivity = np.nanmean(sensitivity_per_class) # Ignore NaN

        # Overall Pixel Accuracy
        total_correct_pixels = tp.sum()
        total_pixels = self._conf_matrix.sum()
        overall_pixel_accuracy = total_correct_pixels / (total_pixels + epsilon) if total_pixels > 0 else 0.0

        results = OrderedDict()
        results["mIoU"] = float(mean_iou * 100) 
        results["pixel_accuracy"] = float(overall_pixel_accuracy * 100)
        results["macro_sensitivity"] = float(macro_sensitivity * 100) # Add macro_sensitivity

        logger.info(
            "NFCMTL Segmentation Evaluator: \n"
            "\n"
        )
        logger.info(f"\nEvaluation Results for {self._dataset_name}:")
        logger.info(f"Mean IoU (mIoU): {results['mIoU']:.2f}%")
        logger.info(f"Overall Pixel Accuracy: {results['pixel_accuracy']:.2f}%")
        logger.info(f"Macro-Averaged Sensitivity (Recall): {results['macro_sensitivity']:.2f}%") # Log macro_sensitivity
        
        logger.info("Per-Class Metrics:")
        for i, class_name in enumerate(self._class_names):
            iou_val = float(iou_per_class[i] * 100)
            sensitivity_val = float(sensitivity_per_class[i] * 100) if (tp[i] + fn[i]) > 0 else 0.0
            
            results[f"IoU-{class_name}"] = iou_val
            results[f"sensitivity-{class_name}"] = sensitivity_val # Changed from accuracy to sensitivity
            
            logger.info(f"  Class: {class_name.ljust(20)} IoU: {iou_val:.2f}%\tSensitivity (Recall): {sensitivity_val:.2f}%")
        
        return results
