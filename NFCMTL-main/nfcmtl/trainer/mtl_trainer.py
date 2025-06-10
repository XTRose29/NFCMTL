from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import hooks
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.solver.build import get_default_optimizer_params
from detectron2.utils import comm
import torch
import torch.nn as nn
import time
import logging
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf # Used by lazy configs

from detectron2.engine.defaults import DefaultTrainer, create_ddp_model
from detectron2.utils.events import EventStorage, get_event_storage
# No CfgNode needed for lazy config version

logger = logging.getLogger(__name__)

class UncertaintyTrainer(DefaultTrainer):
    """
    A custom Detectron2 trainer that applies uncertainty weighting to specified loss components,
    designed to be used with the lazy configuration system.

    This trainer learns task-dependent weights (log_sigma_sq) for different losses,
    allowing the model to balance their contributions automatically.
    The total loss is formulated as:
    Sum_i (0.5 * exp(-log_sigma_sq_i) * RawLoss_i + 0.5 * log_sigma_sq_i)
    for each task 'i' that is being weighted.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (DictConfig): The Detectron2 configuration object, fully resolved
                           from a lazy config .py file.
                           Expected to have `cfg.model.uncertainty_loss` attribute
                           with `weighted_keys` and `uncertainty_lr_multiplier`.
        """
        # Initialize DefaultTrainer, which builds the model, optimizer, dataloader, etc.
        # DefaultTrainer.__init__ calls self.build_model, self.build_optimizer, etc.
        super().__init__(cfg) # cfg here is the fully resolved DictConfig

        # Access uncertainty configuration directly from the cfg object
        # Using getattr to provide default values if not specified, making them optional
        self.uncertainty_config = getattr(cfg, "uncertainty_loss_settings", None)
        
        self.weighted_loss_keys = []
        self.uncertainty_lr_multiplier = 1.0
        self.log_sigma_sq_params = None # Initialize as None

        if self.uncertainty_config:
            self.weighted_loss_keys = getattr(self.uncertainty_config, "weighted_keys", [])
            self.uncertainty_lr_multiplier = getattr(self.uncertainty_config, "uncertainty_lr_multiplier", 1.0)
        else:
            logger.warning(
                "UncertaintyTrainer: `cfg.model.uncertainty_loss` not found. "
                "The trainer will behave like DefaultTrainer."
            )
            return

        if not self.weighted_loss_keys:
            logger.warning(
                "UncertaintyTrainer: No loss keys specified for uncertainty weighting "
                "(cfg.model.uncertainty_loss.weighted_keys is empty or not defined). "
                "The trainer will behave like DefaultTrainer."
            )
            return

        # Initialize learnable parameters for log(sigma_i^2) for each specified loss key.
        # These parameters must be on the same device as the model.
        device = self.model.device 
        self.log_sigma_sq_params = nn.ParameterDict()

        for key in self.weighted_loss_keys:
            # Initialize log_sigma_sq, typically to 0.0.
            self.log_sigma_sq_params[key] = nn.Parameter(torch.zeros(1, device=device))
        
        # Add these new learnable parameters to the optimizer.
        # self.optimizer is created by super().__init__(cfg) via self.build_optimizer(cfg)
        uncertainty_params_list = list(self.log_sigma_sq_params.values())
        
        if uncertainty_params_list: # Ensure there are parameters to add
            # Ensure optimizer and its param_groups are available
            if not hasattr(self, 'optimizer') or not self.optimizer.param_groups:
                logger.error("UncertaintyTrainer: Optimizer not found or not properly initialized by DefaultTrainer.")
                return

            # Determine base LR for uncertainty params.
            # It's safer to access the LR from the first param group if it exists.
            base_lr_for_uncertainty = self.optimizer.param_groups[0]['lr'] * self.uncertainty_lr_multiplier
            
            self.optimizer.add_param_group({
                "params": uncertainty_params_list,
                "lr": base_lr_for_uncertainty,
                "name": "uncertainty_log_sigmas" # Name for this parameter group
            })
            logger.info(
                f"UncertaintyTrainer: Added uncertainty parameters for keys: "
                f"{list(self.log_sigma_sq_params.keys())} to the optimizer."
            )
            logger.info(
                f"UncertaintyTrainer: Learning rate for uncertainty parameters: {base_lr_for_uncertainty:.6f}"
            )

    def run_step(self):
        """
        Overrides DefaultTrainer.run_step() (which is SimpleTrainer.run_step()).
        This method contains the core training logic for a single iteration.
        """
        assert self.model.training, "[UncertaintyTrainer] model was changed to eval mode!"
        start_time = time.perf_counter()

        data = next(self._data_loader_iter)
        data_load_time = time.perf_counter() - start_time

        loss_dict_raw = self.model(data)

        if not self.log_sigma_sq_params or not self.weighted_loss_keys:
            total_loss = sum(loss_dict_raw.values())
            
            metrics_dict = loss_dict_raw.copy()
            metrics_dict["data_time"] = data_load_time
            
            # Use SimpleTrainer's method to log to EventStorage.
            # This is a simplified version of _write_metrics from SimpleTrainer
            storage = get_event_storage()
            storage.put_scalar("total_loss", total_loss.item())
            for k, v in metrics_dict.items():
                try:
                    storage.put_scalar(k, v.item())
                except AttributeError: # If v is already a float/int
                    storage.put_scalar(k, v)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            return

        total_weighted_loss = torch.tensor(0.0, device=self.model.device)
        current_iter_metrics = OrderedDict() 
        unweighted_sum_for_logging = 0.0

        for key, raw_loss_tensor in loss_dict_raw.items():
            unweighted_sum_for_logging += raw_loss_tensor.item()
            if key in self.log_sigma_sq_params:
                log_sigma_sq_param = self.log_sigma_sq_params[key]
                precision = torch.exp(-log_sigma_sq_param)
                task_contribution = 0.5 * precision * raw_loss_tensor + 0.5 * log_sigma_sq_param
                total_weighted_loss += task_contribution
                
                current_iter_metrics[key] = raw_loss_tensor.item()
                current_iter_metrics[f"sigma_{key}"] = torch.exp(0.5 * log_sigma_sq_param).item()
                current_iter_metrics[f"log_sigma_sq_{key}"] = log_sigma_sq_param.item()
                current_iter_metrics[f"wloss_{key}"] = task_contribution.item()
            else:
                total_weighted_loss += raw_loss_tensor 
                current_iter_metrics[key] = raw_loss_tensor.item()

        self.optimizer.zero_grad()
        total_weighted_loss.backward()
        self.optimizer.step()

        storage = get_event_storage()
        storage.put_scalar("total_loss", total_weighted_loss.item())
        storage.put_scalar("unweighted_total_loss", unweighted_sum_for_logging)
        for metric_key, metric_value in current_iter_metrics.items():
            storage.put_scalar(metric_key, metric_value)
        storage.put_scalar("data_time", data_load_time, smoothing_hint=False)

# Note: The _write_metrics method is part of SimpleTrainer, which DefaultTrainer uses.
# If you directly inherit SimpleTrainer, you might need to ensure _write_metrics is compatible
# or reimplement the parts of it you need for logging if uncertainty is off.
# For this DefaultTrainer inheritance, the simplified logging within run_step when
# uncertainty is off should suffice.



class UncertaintyTrainerSimp(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but applies uncertainty weighting to specified loss components.
    The model, data_loader, and optimizer are provided pre-built.
    Uncertainty configuration is read from `cfg.uncertainty_loss_settings`.
    """
    def __init__(
        self,
        model, # Already instantiated, potentially DDP-wrapped model
        data_loader, # Already instantiated data loader
        optimizer, # Already instantiated optimizer
        cfg: DictConfig, # Full configuration for uncertainty settings
        # Optional SimpleTrainer args, if needed, though typically not for this use case
        # gather_metric_period=1, 
        # zero_grad_before_forward=False,
    ):
        """
        Args:
            model: The model to train.
            data_loader: The data loader to use.
            optimizer: The optimizer to use.
            cfg: The Detectron2 configuration object, used to retrieve
                 `uncertainty_loss_settings`.
        """
        super().__init__(model, data_loader, optimizer) # Initialize SimpleTrainer base

        self.cfg = cfg # Store cfg for convenience if needed elsewhere, though primarily for init

        # Setup uncertainty parameters
        self.uncertainty_config = getattr(cfg, "uncertainty_loss_settings", None)
        
        self.weighted_loss_keys = []
        self.uncertainty_lr_multiplier = 1.0
        self.log_sigma_sq_params = None # Initialize as None

        if self.uncertainty_config:
            if isinstance(self.uncertainty_config, (dict, DictConfig)):
                self.weighted_loss_keys = self.uncertainty_config.get("weighted_keys", [])
                self.uncertainty_lr_multiplier = self.uncertainty_config.get("uncertainty_lr_multiplier", 1.0)
            else:
                 logger.error(
                    "UncertaintyTrainer: `cfg.uncertainty_loss_settings` is not a dictionary-like object."
                )
                 # Trainer will run like SimpleTrainer without uncertainty if config is malformed
        else:
            logger.warning(
                "UncertaintyTrainer: `cfg.uncertainty_loss_settings` not found. "
                "The trainer will behave like SimpleTrainer without uncertainty weighting."
            )
        
        if self.weighted_loss_keys:
            # Device should be the same as the model's device
            # self.model is set by SimpleTrainer.__init__
            device = next(self.model.parameters()).device 
            self.log_sigma_sq_params = nn.ParameterDict()

            for key in self.weighted_loss_keys:
                # Initialize log_sigma_sq, typically to 0.0.
                self.log_sigma_sq_params[key] = nn.Parameter(torch.zeros(1, device=device))
            
            # Add these new learnable parameters to the optimizer.
            # self.optimizer is set by SimpleTrainer.__init__
            uncertainty_params_list = list(self.log_sigma_sq_params.values())
            
            if uncertainty_params_list:
                if not hasattr(self, 'optimizer') or not self.optimizer.param_groups:
                    logger.error("UncertaintyTrainer: Optimizer not found or not properly initialized by SimpleTrainer.")
                else:
                    # Determine base LR for uncertainty params from the first param group of the optimizer.
                    base_lr_for_uncertainty = self.optimizer.param_groups[0]['lr'] * self.uncertainty_lr_multiplier
                    
                    self.optimizer.add_param_group({
                        "params": uncertainty_params_list,
                        "lr": base_lr_for_uncertainty,
                        "name": "uncertainty_log_sigmas" 
                    })
                    logger.info(
                        f"UncertaintyTrainer: Added uncertainty parameters for keys: "
                        f"{list(self.log_sigma_sq_params.keys())} to the optimizer."
                    )
                    logger.info(
                        f"UncertaintyTrainer: Learning rate for uncertainty parameters: {base_lr_for_uncertainty:.6f}"
                    )
        else:
             logger.warning(
                "UncertaintyTrainer: No loss keys specified for uncertainty weighting. "
                "The trainer will behave like SimpleTrainer."
            )

    def run_step(self):
        """
        Implement the training logic with uncertainty weighting for a single iteration.
        This method is called by the external training loop.
        """
        assert self.model.training, "[UncertaintyTrainer] model was changed to eval mode!"
        start_time = time.perf_counter()

        # _data_loader_iter is an attribute from SimpleTrainer, initialized by its train() method,
        # or needs to be managed by the external loop if super().train() isn't called.
        # Assuming the external loop handles providing data or manages the iterator.
        # If using SimpleTrainer's iteration logic, self._data_loader_iter will be used.
        # For now, let's assume `data = next(self._data_loader_iter)` is standard.
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            # This logic is from SimpleTrainer.train() when the iterator is exhausted.
            # The external loop or a custom train method in a derived class would handle this.
            # If this trainer's run_step is called directly in a loop, the loop needs to manage this.
            # For now, we assume the iterator is valid for one step.
            # If your main.py uses a loop that calls trainer.run_step(), it should handle iterator reset.
            # Let's keep SimpleTrainer's re-init logic for robustness if its internal iter is used.
            logger.info("[UncertaintyTrainer] Re-initializing data loader iterator in run_step.")
            self._data_loader_iter = iter(self.data_loader) # self.data_loader from SimpleTrainer
            data = next(self._data_loader_iter)


        data_load_time = time.perf_counter() - start_time

        # Forward pass to get raw losses from the model (self.model is from SimpleTrainer)
        loss_dict_raw = self.model(data)

        # If uncertainty weighting is not configured or no keys, behave like SimpleTrainer's loss sum.
        if not self.log_sigma_sq_params or not self.weighted_loss_keys:
            if isinstance(loss_dict_raw, torch.Tensor): # Single loss tensor
                total_loss = loss_dict_raw
                loss_dict_raw_for_logging = {"total_loss": total_loss.item()}
            else: # Dictionary of losses
                total_loss = sum(v for v in loss_dict_raw.values() if torch.is_tensor(v) and v.requires_grad) # Sum only loss tensors
                loss_dict_raw_for_logging = {k: v.item() for k, v in loss_dict_raw.items() if torch.is_tensor(v)}

            # Standard logging for SimpleTrainer-like behavior
            # _write_metrics is a method of SimpleTrainer.
            # It expects loss_dict (raw values) and data_time.
            # It also uses self.iter, which should be set by the external loop.
            if hasattr(self, '_write_metrics'):
                 self._write_metrics(loss_dict_raw_for_logging, data_load_time) # iter is used from self.iter
            else: # Fallback if _write_metrics is somehow not available (should be)
                storage = get_event_storage()
                storage.put_scalar("total_loss", total_loss.item())
                for k, v in loss_dict_raw_for_logging.items():
                    storage.put_scalar(k, v)
                storage.put_scalar("data_time", data_load_time)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step() # self.optimizer from SimpleTrainer
            return # Exit early

        # --- Uncertainty Weighting Logic ---
        total_weighted_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        current_iter_metrics_for_logging = OrderedDict() 
        unweighted_sum_for_logging = 0.0

        for key, raw_loss_tensor in loss_dict_raw.items():
            if not (torch.is_tensor(raw_loss_tensor) and raw_loss_tensor.requires_grad): # Skip non-loss tensors
                if torch.is_tensor(raw_loss_tensor): # Log non-loss tensors if they are tensors
                    current_iter_metrics_for_logging[key] = raw_loss_tensor.item()
                continue

            unweighted_sum_for_logging += raw_loss_tensor.item()
            if key in self.log_sigma_sq_params:
                log_sigma_sq_param = self.log_sigma_sq_params[key]
                precision = torch.exp(-log_sigma_sq_param)
                # Ensure task_contribution is a scalar by summing if log_sigma_sq_param was (1,)
                task_contribution = (0.5 * precision * raw_loss_tensor + 0.5 * log_sigma_sq_param).sum() 
                total_weighted_loss += task_contribution
                
                current_iter_metrics_for_logging[key] = raw_loss_tensor.item() # Original (raw) loss
                current_iter_metrics_for_logging[f"sigma_{key}"] = torch.exp(0.5 * log_sigma_sq_param).item()
                current_iter_metrics_for_logging[f"log_sigma_sq_{key}"] = log_sigma_sq_param.item()
                current_iter_metrics_for_logging[f"wloss_{key}"] = task_contribution.item() # Weighted contribution
            else:
                # If a loss component is not weighted, add it directly.
                total_weighted_loss += raw_loss_tensor 
                current_iter_metrics_for_logging[key] = raw_loss_tensor.item() # Log its raw value

        self.optimizer.zero_grad()
        total_weighted_loss.backward()
        self.optimizer.step()

        # --- Logging via Detectron2's EventStorage ---
        # This part is usually handled by SimpleTrainer._write_metrics.
        # We are doing it explicitly here for the uncertainty case.
        # The external loop should set self.iter.
        storage = get_event_storage()
        storage.put_scalar("total_loss", total_weighted_loss.item()) # This is the primary loss for display
        storage.put_scalar("unweighted_total_loss", unweighted_sum_for_logging)
        for metric_key, metric_value in current_iter_metrics_for_logging.items():
            storage.put_scalar(metric_key, metric_value)
        storage.put_scalar("data_time", data_load_time, smoothing_hint=False)
        # LR is typically logged by LRScheduler hook, which observes the optimizer.

    def state_dict(self):
        ret = super().state_dict() # Gets optimizer state and iter from SimpleTrainer
        # Add log_sigma_sq_params to checkpoint
        if self.log_sigma_sq_params:
            # Detach and move to CPU for state_dict
            ret["log_sigma_sq_params"] = {
                k: v.detach().cpu() for k, v in self.log_sigma_sq_params.items()
            }
        return ret

    def load_state_dict(self, state_dict):
        # SimpleTrainer's load_state_dict handles optimizer and iter
        # We need to remove "log_sigma_sq_params" before calling super if it's not expected by SimpleTrainer
        # However, SimpleTrainer's load_state_dict is basic, so it's fine to call it first.
        
        # Load log_sigma_sq_params if present
        if "log_sigma_sq_params" in state_dict:
            if self.log_sigma_sq_params: # Ensure current trainer is configured for uncertainty
                logger.info("Loading log_sigma_sq_params from checkpoint.")
                loaded_log_sigmas = state_dict["log_sigma_sq_params"]
                for k, v_loaded in loaded_log_sigmas.items():
                    if k in self.log_sigma_sq_params:
                        v_current = self.log_sigma_sq_params[k]
                        # Ensure it's loaded to the correct device and type
                        try:
                            with torch.no_grad():
                                v_current.copy_(v_loaded.to(device=v_current.device, dtype=v_current.dtype))
                        except Exception as e:
                             logger.error(f"Error loading log_sigma_sq_param '{k}': {e}. Skipping.")
                    else:
                        logger.warning(f"log_sigma_sq_param '{k}' from checkpoint not found in current trainer setup.")
            else:
                logger.warning("Checkpoint contains 'log_sigma_sq_params' but current trainer is not configured for them.")
        
        # Remove custom keys before calling super to avoid issues if SimpleTrainer is strict,
        # though SimpleTrainer's load_state_dict is usually fine.
        # For safety, create a copy for super.
        simple_trainer_state = {k: v for k, v in state_dict.items() if k != "log_sigma_sq_params"}
        super().load_state_dict(simple_trainer_state)
