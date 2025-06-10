import numpy as np
import pycocotools.mask as mask_util  # For mapper
import torch

from detectron2.data import (
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,  # We might need this if not using instantiate for loader directly
)


class CustomDatasetMapperWithSemSeg(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        # Assuming 'cfg' is the global config object
        # For test, we get dataset name from cfg.DATASETS.TEST[0] or similar
        # For train, cfg.DATASETS.TRAIN[0]
        dataset_names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
        if isinstance(dataset_names, str): # handle single string
            dataset_names = [dataset_names]

        if not dataset_names:
            raise ValueError("Dataset names not found in CFG for mapper.")
            
        self.meta = MetadataCatalog.get(dataset_names[0]) # Use first dataset for metadata
        
        # Ensure ignore_label is available
        try:
            self.ignore_label = self.meta.ignore_label
        except AttributeError:
            self._logger.warning(
                f"'ignore_label' not found in metadata for {dataset_names[0]}. "
                "Defaulting ignore_label to 255 for semantic mask generation."
            )
            self.ignore_label = 255 # Make sure this is consistent with evaluator

        # Ensure thing_dataset_id_to_contiguous_id is available if using 'thing' classes
        try:
            self.thing_contiguous_id_map = self.meta.thing_dataset_id_to_contiguous_id
        except AttributeError:
            self._logger.warning(
                f"'thing_dataset_id_to_contiguous_id' not found in metadata for {dataset_names[0]}. "
                "Assuming category_ids are already contiguous or not needed for semantic mapping."
            )
            self.thing_contiguous_id_map = None # Handle this case below

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        if "instances" not in dataset_dict and "annotations" not in dataset_dict:
            # If no annotations, create an empty sem_seg_gt or one filled with ignore_label
            h = dataset_dict["height"]
            w = dataset_dict["width"]
            sem_seg_gt = np.full((h, w), self.ignore_label, dtype=np.uint8)
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
            return dataset_dict

        # If "instances" are present (they should be after super().__call__ if annotations exist)
        if "instances" in dataset_dict:
            instances = dataset_dict["instances"] # This is an Instances object
            h, w = instances.image_size

            sem_seg_gt = np.full((h, w), self.ignore_label, dtype=np.uint8)

            if instances.has("gt_masks") and instances.has("gt_classes"):
                gt_masks = instances.gt_masks # This is a BitMasks object or PolygonMasks
                gt_classes = instances.gt_classes # These should be contiguous IDs

                for i in range(len(instances)):
                    class_id = gt_classes[i].item() # This is the contiguous ID
                    mask = gt_masks.tensor[i].cpu().numpy() # Get i-th mask as HxW numpy array
                    sem_seg_gt[mask > 0] = class_id
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        
        # Fallback: if 'annotations' are still there and not processed to 'instances' by base mapper
        elif "annotations" in dataset_dict and dataset_dict["annotations"]:
            h = dataset_dict["height"]
            w = dataset_dict["width"]
            sem_seg_gt = np.full((h, w), self.ignore_label, dtype=np.uint8)
            
            for ann in dataset_dict["annotations"]:
                if ann.get("iscrowd", 0):
                    continue
                if "segmentation" in ann:
                    coco_class_id = ann["category_id"]

                    if self.thing_contiguous_id_map:
                        class_id = self.thing_contiguous_id_map.get(coco_class_id, self.ignore_label)
                    else: # Assume coco_class_id is already the target class_id (0-indexed)
                        class_id = coco_class_id
                    
                    if class_id == self.ignore_label:
                        continue

                    if isinstance(ann["segmentation"], list): # Polygon
                        rles = mask_util.frPyObjects(ann["segmentation"], h, w)
                        rle = mask_util.merge(rles)
                    elif isinstance(ann["segmentation"], dict) and "counts" in ann["segmentation"]: # RLE
                        rle = ann["segmentation"]
                    else:
                        continue 
                    mask = mask_util.decode(rle) # Binary mask
                    sem_seg_gt[mask > 0] = class_id
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            # If no instances and no annotations, create empty/ignored sem_seg
            h = dataset_dict["height"]
            w = dataset_dict["width"]
            sem_seg_gt = np.full((h, w), self.ignore_label, dtype=np.uint8)
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
            
        return dataset_dict
