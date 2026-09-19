import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import pynvml
from torch import Tensor
import cv2
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
from typing import Any, Callable, Dict, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt           

from utils.utils import set_seed

from model.handlers.data_handler import DataHandler
from model.handlers.config_handler import ConfigHandler
from model.handlers.model_handler import ModelHandler
from model.handlers.parallel_handler import ParallelHandler
import torch.nn.functional as F
from itertools import cycle
from einops import rearrange

from model.losses.losses import embedding_loss, embedding_loss_teacher_student, feature_distillation_loss
from utils.affinity import multi_offset

class Trainer:
    """Trainer class to handle the training, validation, and testing of a model"""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.config = ConfigHandler(config).config
        set_seed(self.config.get('random_seed', 321))
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = min(torch.cuda.device_count(), self.config.get('num_gpus', 1))
        
        # if self.num_gpus > 1:
        #     if self.config['batch_size'] % self.num_gpus != 0:
        #         raise ValueError("Batch size must be divisible by the number of GPUs.")
        # print(f"Using {self.num_gpus} GPUs for training.")

        # Data loaders
        self.train_loader, self.val_loader = DataHandler(self.config)._get_train_dataloaders()

        # Initialize student and teacher models
        self.teacher_model_handler = ModelHandler(self.config, self.device, is_teacher=True)
        self.student_model_handler = ModelHandler(self.config, self.device, is_teacher=False)

        # Single/Mixed precision training
        self.dtype = self.map_precision(self.config.get('data_precision', 'float32'))

        # Load pretrained teacher model and freeze it
        if not self.config['resume']:
            self.load_pretrainedModel_teacher_and_student()
            self.freeze_model(self.teacher_model_handler.model)

        # Apply parallelization if necessary
        self.teacher_model_handler.model = ParallelHandler.apply_parallel(self.teacher_model_handler.model, self.num_gpus)
        self.student_model_handler.model = ParallelHandler.apply_parallel(self.student_model_handler.model, self.num_gpus)
        
        # Casting teacher and student models to the specified dtype
        self.force_model_to_dtype(self.teacher_model_handler.model, self.dtype)
        self.force_model_to_dtype(self.student_model_handler.model, self.dtype)



        # Optimizer 
        #self.optimizer_teacher = self.teacher_model_handler.optimizer
        #self.optimizer_student = self.student_model_handler.optimizer
        self.optimizer = self.student_model_handler.optimizer
        # self.cast_optimizer_to_dtype(self.student_model_handler.optimizer, self.dtype)

        # Scheduler
        #self.scheduler_teacher = self.teacher_model_handler.scheduler   
        #self.scheduler_student = self.student_model_handler.scheduler   
        self.scheduler = self.student_model_handler.scheduler   

        # EMA decay for teacher update
        self.ema_decay = self.config.get('ema_decay', 0.999)  # EMA decay factor

        # Gradient clipping for stability
        self.grad_clip_max_norm = self.config.get('grad_clip_max_norm', 1.0)
        self.grad_clip_enabled = self.config.get('grad_clip_enabled', True)

        # Feature distillation weight (for embedding consistency between teacher and student)
        self.feature_distillation_weight = self.config.get('feature_distillation_weight', 0.01)
        self.feature_distillation_enabled = self.config.get('feature_distillation_enabled', True)

        # NaN detection and stability monitoring
        self.nan_skip_count = 0
        self.max_nan_skips_per_epoch = self.config.get('max_nan_skips_per_epoch', 10)
        self.loss_scale_factor = self.config.get('loss_scale_factor', 1.0)  # For loss scaling if needed

        # Initialize padding_size
        self.padding_size = self.config.get('padding_size', 0)

        # Initialize offsets
        self.offsets = multi_offset(shifts=self.config['shifts_affinity'], neighbor=self.config['neighbor_affinity'])

        # Training state
        self.best_val_loss = float('inf')
        self.prev_lr = None
        self.start_epoch = 0  
        self.end_epoch = self.config['epochs']

        # Initialize GradScaler if mixed precision is used
        # self.scaler = torch.cuda.amp.GradScaler() if not self.single_precision else None
        # self.scaler = torch.amp.GradScaler(device=self.device.type) if self.dtype != torch.float32 else None

        if self.dtype == torch.float16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Resume training or fine-tuning or training from scratch
        if self.config['resume']:
            checkpoint_type = self.config.get('checkpoint_type', 'latest')
            checkpoint_path = self.get_checkpoint_path(checkpoint_type)
            self.load_checkpoint(checkpoint_path)
            self.prev_lr = self.get_lr()
            print(f"Resuming training from epoch {self.start_epoch + 1} with learning rate: {self.prev_lr}")
        elif self.config['fine_tune']:
            raise NotImplementedError("Fine-tuning functionality is not yet implemented.")
        else:
            # TensorBoard writer
            start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(self.config.get('log_dir', './logs'), start_time)
            self.writer = SummaryWriter(log_dir=log_dir)

        self.cumulative_losses_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'total': 0.0}
        self.cumulative_losses_weakly_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'feature_distillation': 0.0, 'total': 0.0}

        # Gradient norm tracking for monitoring
        self.cumulative_grad_norm = 0.0


    def force_model_to_dtype(self, model: nn.Module, target_dtype: torch.dtype) -> None:
        """
        Casts the model to the target dtype while keeping BatchNorm layers in float32
        for numerical stability in mixed-precision training.
        """
        model.to(dtype=target_dtype)

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()

        print(f"Model cast to {target_dtype}; BatchNorm kept in float32.")



    def cast_weights(self, state_dict, dtype):
        """Cast all weights to the specified dtype."""
        for name, param in state_dict.items():
            state_dict[name] = param.to(dtype=dtype)
        return state_dict



    @staticmethod
    def map_precision(data_precision: str):
        """Map string precision to PyTorch's data types."""
        precision_mapping = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        if data_precision not in precision_mapping:
            raise ValueError(f"Unsupported data precision: {data_precision}. Choose from {list(precision_mapping.keys())}.")
        return precision_mapping[data_precision]
    
    def get_checkpoint_path(self, checkpoint_type):
        """
        Get the path to the specified checkpoint type.

        Parameters:
        - checkpoint_type (str): Type of checkpoint ('latest', 'best', or a custom path).

        Returns:
        - str: Path to the checkpoint file.
        """
        if checkpoint_type in ['latest', 'best']:
            return os.path.join(self.config['checkpoint_dir'], f'checkpoint_{checkpoint_type}.pth')
        else:
            return checkpoint_type

    def save_checkpoint(self, epoch, is_best=False, is_specific=False):
        state = {
            'epoch': epoch + 1,
            #'state_dict': self.model_handler.model.module.state_dict() if self.num_gpus > 1 else self.model_handler.model.state_dict(),
            'student_state_dict': self.student_model_handler.model.module.state_dict() if self.num_gpus > 1 else self.student_model_handler.model.state_dict(),
            'teacher_state_dict': self.teacher_model_handler.model.module.state_dict() if self.num_gpus > 1 else self.teacher_model_handler.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'log_dir': self.writer.log_dir
        }

        if is_specific and not is_best:
            specific_filename = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(state, specific_filename)
            print(f"Checkpoint saved at epoch {epoch + 1} to {specific_filename}")
        
        if not is_specific and not is_best:
            latest_filename = os.path.join(self.config['checkpoint_dir'], 'checkpoint_latest.pth')
            torch.save(state, latest_filename)
        
        if is_best:
            best_filename = os.path.join(self.config['checkpoint_dir'], 'checkpoint_best.pth')
            torch.save(state, best_filename)
            print(f"★ Best model saved at epoch {epoch + 1} to '{best_filename}'")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and set model and state-related params."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load student model state
        #student_state_dict = ParallelHandler.adjust_state_dict(checkpoint['student_state_dict'], self.student_model_handler.model, self.num_gpus)
        student_state_dict = ParallelHandler.adjust_state_dict(checkpoint['student_state_dict'], self.num_gpus)
        self.student_model_handler.model.load_state_dict(student_state_dict)

        # Load teacher model state
        #teacher_state_dict = ParallelHandler.adjust_state_dict(checkpoint['teacher_state_dict'], self.teacher_model_handler.model, self.num_gpus)
        teacher_state_dict = ParallelHandler.adjust_state_dict(checkpoint['teacher_state_dict'], self.num_gpus)
        self.teacher_model_handler.model.load_state_dict(teacher_state_dict)

        # Load optimizer and scheduler state for student model
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch']
        self.end_epoch = self.start_epoch + self.config['epochs']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.writer = SummaryWriter(log_dir=checkpoint['log_dir'])
        print(f"Loaded checkpoint from epoch {self.start_epoch}")


    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']


    def load_pretrainedModel_teacher_and_student(self):
        """Load the pretrained weights for the teacher model and freeze the teacher."""
        # Load the checkpoint from the file
        checkpoint_teacher = torch.load(self.config['pretrained_teacher_model_dir'], map_location=self.device, weights_only=False)
        checkpoint_student = torch.load(self.config['pretrained_student_model_dir'], map_location=self.device, weights_only=False)

        # Adjust the state_dict keys by removing 'module.' prefix if needed
        #adjusted_state_dict_teacher = self.remove_module_prefix(checkpoint_teacher['state_dict'])
        #adjusted_state_dict_student = self.remove_module_prefix(checkpoint_student['state_dict'])


        # Adjust the state_dict and cast sensitive layers to float32
        # adjusted_state_dict_teacher = self.cast_weights(self.remove_module_prefix(checkpoint_teacher['state_dict']), dtype=self.dtype)
        # adjusted_state_dict_student = self.cast_weights(self.remove_module_prefix(checkpoint_student['state_dict']), dtype=self.dtype)

        adjusted_state_dict_teacher = self.remove_module_prefix(checkpoint_teacher['state_dict'])
        adjusted_state_dict_student = self.remove_module_prefix(checkpoint_student['state_dict'])


        # Load the adjusted state dict into the teacher model
        self.teacher_model_handler.model.load_state_dict(adjusted_state_dict_teacher)
        print("Loaded pre-trained teacher model from", self.config['pretrained_teacher_model_dir'])

        self.student_model_handler.model.load_state_dict(adjusted_state_dict_student)
        print("Loaded pre-trained student model from", self.config['pretrained_student_model_dir'])
        

    def remove_module_prefix(self, state_dict):
        """Remove 'module.' prefix from keys in state_dict if present."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]  # Remove 'module.' prefix
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        return new_state_dict
    
    def freeze_model(self, model):
        """Freeze the entire model (teacher) so that its weights won't be updated."""
        for param in model.parameters():
            param.requires_grad = False
        print("Teacher model has been frozen.")


    def update_teacher_model_ema(self, student_model, teacher_model, decay):
        """Update the teacher model using EMA from the student model.

        Syncs both learnable parameters (weight/bias) and buffers
        (e.g. BatchNorm running_mean/running_var) to keep teacher weights
        consistent with the statistics used at inference time.
        """
        with torch.no_grad():
            student_params = dict(student_model.named_parameters())
            teacher_params = dict(teacher_model.named_parameters())

            for name in teacher_params.keys():
                if torch.isnan(student_params[name].data).any():
                    print(f"WARNING: NaN detected in student param '{name}', skipping EMA update for this param")
                    continue
                teacher_params[name].data.mul_(decay).add_(student_params[name].data, alpha=1.0 - decay)

            # Skip buffer EMA: with batch_size=2 the student's BN running stats are too noisy;
            # blending them into the teacher polluted teacher BN and degraded teacher metric.
            # Teacher keeps its pretrained BN stats.
            # student_buffers = dict(student_model.named_buffers())
            # teacher_buffers = dict(teacher_model.named_buffers())
            # for name, t_buf in teacher_buffers.items():
            #     if name not in student_buffers:
            #         continue
            #     s_buf = student_buffers[name]
            #     # Integer counters like num_batches_tracked: copy directly
            #     if not t_buf.dtype.is_floating_point:
            #         t_buf.data.copy_(s_buf.data)
            #         continue
            #     if torch.isnan(s_buf.data).any():
            #         print(f"WARNING: NaN detected in student buffer '{name}', skipping EMA update for this buffer")
            #         continue
            #     t_buf.data.mul_(decay).add_(s_buf.data, alpha=1.0 - decay)

    def check_for_nan(self, tensor, name="tensor"):
        """Check if a tensor contains NaN or Inf values."""
        if tensor is None:
            return False
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"WARNING: {'NaN' if has_nan else 'Inf'} detected in {name}")
            return True
        return False

    def compute_gradient_norm(self, model):
        """Compute the total gradient norm for all parameters."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def clip_gradients(self, model, max_norm):
        """Clip gradients and return the gradient norm before clipping."""
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def safe_loss_backward(self, loss, batch_idx, epoch):
        """
        Safely perform backward pass with NaN checking.
        Returns True if backward was successful, False if skipped due to NaN.
        """
        # Check for NaN in loss
        if self.check_for_nan(loss, f"loss at epoch {epoch}, batch {batch_idx}"):
            self.nan_skip_count += 1
            if self.nan_skip_count > self.max_nan_skips_per_epoch:
                raise RuntimeError(f"Too many NaN losses ({self.nan_skip_count}) in epoch {epoch}. Training unstable.")
            print(f"Skipping batch {batch_idx} due to NaN loss (skip count: {self.nan_skip_count})")
            self.optimizer.zero_grad()  # Clear any partial gradients
            return False

        # Perform backward pass
        if self.scaler is not None:
            # float16 path
            self.scaler.scale(loss).backward()
        else:
            # float32 or bfloat16 path
            loss.backward()

        return True


    def log_images(self, epoch, batch_idx,
                inp_img_supervised,        pred_mask_supervised,        gt_mask_supervised,      pred_affinity_sup, gt_labels_affs_supervised,
                inp_img_weakly_supervised, pred_mask_weakly_supervised, gt_mask_weakly_supervised,
                mode='train'):
        """
        Logs images for both supervised and weakly-supervised data to the specified directory.
        """
        # Define the directory to save images
        save_dir = os.path.join(self.config['log_dir'], mode, f'{mode}_results_epoch_{epoch + 1}_batch_{batch_idx}')
        os.makedirs(save_dir, exist_ok=True)

        # Log for supervised data
        self._log_individual_images(
            save_dir, inp_img_supervised, pred_mask_supervised, gt_mask_supervised, pred_affinity_sup, gt_labels_affs_supervised, "supervised"
        )

        # Log for weakly-supervised data (without affinity visualizations)
        self._log_individual_images(
            save_dir, inp_img_weakly_supervised, pred_mask_weakly_supervised, gt_mask_weakly_supervised, None, None, "weakly_supervised", log_affinities=False
        )


    def _log_individual_images(self, save_dir, inp_img, pred_mask, gt_mask, pred_affinity, gt_labels_affs, label, log_affinities=True):
        """
        Logs individual images including input image, probability maps, segmentation maps, and ground truth.

        Args:
            save_dir (str): The directory to save images.
            inp_img (torch.Tensor): Input image tensor.
            pred_mask (torch.Tensor): Predicted mask from the model.
            gt_mask (torch.Tensor): Ground truth segmentation mask.
            pred_affinity (torch.Tensor): Predicted affinity map.
            gt_labels_affs (torch.Tensor): Ground truth affinity map.
            label (str): Label for saving image results.
            log_affinities (bool): Whether to log affinity maps or not.
        """
        def process_map(map):
            apply_remove_small_objects = self.config.get('remove_small_objects', True)
            min_size = self.config.get('min_object_size', 150)
            if apply_remove_small_objects:
                map = self.remove_small_objects(map, min_size)
            fill_holes = self.config.get('fill_holes', False)
            if fill_holes:
                map = self.fill_holes(map)
            apply_morphology = self.config.get('apply_morphology', True)
            if apply_morphology:
                kernel_size = self.config.get('morph_kernel_size', (3, 3))
                kernel = np.ones(kernel_size, np.uint8)
                map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
            return map

        # Create a save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Convert predicted logits to probabilities using softmax
        prob_map = F.softmax(pred_mask, dim=1)

        # Detach tensors and convert to numpy arrays with explicit float32 conversion
        inp_img_np = inp_img[0, 0].to(torch.float32).detach().cpu().numpy()  # First batch, first channel
        prob_map_np = prob_map[0].to(torch.float32).detach().cpu().numpy()   # First batch
        gt_mask_np = gt_mask[0, 0].to(torch.float32).detach().cpu().numpy()  # First batch, first channel
        pred_affinity_np = pred_affinity[0].to(torch.float32).detach().cpu().numpy() if pred_affinity is not None else None
        gt_affinity_np = gt_labels_affs[0].to(torch.float32).detach().cpu().numpy() if gt_labels_affs is not None else None

        # Generate the argmax segmentation map
        argmax_segmentation_map = np.argmax(pred_mask[0].to(torch.float32).detach().cpu().numpy(), axis=0).astype(np.uint8)
        argmax_segmentation_map = process_map(argmax_segmentation_map)

        # Threshold maps for visualization at multiple thresholds
        thresholds = [0.25, 0.35, 0.4, 0.45]
        threshold_maps = [(prob_map_np[1] > t).astype(np.uint8) for t in thresholds]
        threshold_maps_cleaned = [process_map(thresh_map) for thresh_map in threshold_maps]

        # Create a figure with 3 rows and 4 columns for images
        fig, axes = plt.subplots(3, 4, figsize=(28, 15))
        titles = ['Input Image', 'Probability Map (Class 0)', 'Probability Map (Class 1)', 'Ground Truth Mask',
                'Class 0 Histogram', 'Class 1 Histogram', 'Argmax Segmentation', 'Ground Truth Affinity'] + \
                [f'Threshold {t}' for t in thresholds]  # Add titles for threshold maps

        # Prepare images for display
        images = [
            inp_img_np,                   # Input image
            prob_map_np[0],               # Probability map for Class 0
            prob_map_np[1],               # Probability map for Class 1
            gt_mask_np,                   # Ground truth mask
            None,                         # Placeholder for Class 0 histogram
            None,                         # Placeholder for Class 1 histogram
            argmax_segmentation_map,      # Argmax segmentation map
            gt_affinity_np[0] if gt_affinity_np is not None else None  # Ground truth affinity
        ] + threshold_maps_cleaned  # Add cleaned maps for threshold visualizations

        # Plot histograms for Class 0 and Class 1 probabilities
        for idx, (ax, title) in enumerate(zip(axes.flatten(), titles)):
            ax.set_title(title)
            ax.axis('off')

            if idx == 4:  # Class 0 histogram
                min_val, max_val = prob_map_np[0].min(), prob_map_np[0].max()
                if np.isfinite(min_val) and np.isfinite(max_val):  # Check if values are finite
                    ax.hist(prob_map_np[0].ravel(), bins=50, range=(min_val, max_val), color='blue', edgecolor='black', alpha=0.7)
                    mean_val, std_val = prob_map_np[0].mean(), prob_map_np[0].std()
                    ax.text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}",
                            transform=ax.transAxes, color='black', fontsize=10,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                else:
                    print(f"Warning: NaN values encountered in class 0 probability map for batch {idx}. Skipping histogram.")

            elif idx == 5:  # Class 1 histogram
                min_val, max_val = prob_map_np[1].min(), prob_map_np[1].max()
                if np.isfinite(min_val) and np.isfinite(max_val):  # Check if values are finite
                    ax.hist(prob_map_np[1].ravel(), bins=50, range=(min_val, max_val), color='orange', edgecolor='black', alpha=0.7)
                    mean_val, std_val = prob_map_np[1].mean(), prob_map_np[1].std()
                    ax.text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}",
                            transform=ax.transAxes, color='black', fontsize=10,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                else:
                    print(f"Warning: NaN values encountered in class 1 probability map for batch {idx}. Skipping histogram.")

            elif images[idx] is not None:  # For images
                ax.imshow(images[idx], cmap='gray')

        # Adjust layout and save figure
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'results_{label}_with_thresholds.png'))
        plt.close(fig)




    @staticmethod
    def remove_small_objects(segmentation_map, min_size):
        """
        Removes small objects from a binary segmentation map.

        Args:
            segmentation_map (np.ndarray): Binary segmentation map.
            min_size (int): Minimum size of objects to keep.

        Returns:
            np.ndarray: Cleaned segmentation map with small objects removed.
        """
        # Label connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmentation_map, connectivity=8)

        # Create an output map for large objects
        output_map = np.zeros_like(segmentation_map)

        # Keep only the large components
        for i in range(1, num_labels):  # Skip the background label (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output_map[labels == i] = 1

        return output_map


    @staticmethod
    def fill_holes(segmentation_map):
        """
        Fills small holes within the segmented objects using flood fill.

        Args:
            segmentation_map (np.ndarray): Binary segmentation map (objects are 1, background is 0).

        Returns:
            np.ndarray: Segmentation map with holes filled.
        """
        # Create an inverted copy of the segmentation map (background as 1, objects as 0)
        flood_fill_map = segmentation_map.copy()
        h, w = flood_fill_map.shape

        # Create a mask for flood filling (needs to be 2 pixels larger than the image)
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Perform flood fill from the top-left corner (considering it as background)
        cv2.floodFill(flood_fill_map, mask, (0, 0), 1)

        # Invert the flood-filled result to get holes filled in the original objects
        flood_fill_map = cv2.bitwise_not(flood_fill_map)

        # Combine with the original segmentation map to fill the holes
        filled_map = segmentation_map | flood_fill_map

        return filled_map


    @staticmethod
    def reshape_batch_dim(input_tensor):
        """
        Reshapes the input tensor from shape [B, N, C, W, H] to [B * N, C, W, H] using einops.rearrange.
        
        Args:
            input_tensor (torch.Tensor): Tensor with shape [B, N, C, W, H] where B is the batch size 
                                        and N is the number of samples per batch.

        Returns:
            reshaped_tensor (torch.Tensor): Tensor with shape [B * N, C, W, H].
        """
        
        # Ensure input tensor has the correct number of dimensions
        if input_tensor.ndimension() != 5:
            raise ValueError(f"Expected 5D input tensor, but got {input_tensor.ndimension()} D tensor.")
        
        # Reshape to [B * N, C, W, H] using einops.rearrange
        reshaped_tensor = rearrange(input_tensor, 'B N C W H -> (B N) C W H')
        
        return reshaped_tensor


    def update_learning_rate(self, min_lr_threshold=1e-4):
        """
        Check and update the learning rate, ensuring it doesn't fall below a minimum threshold.
        If the learning rate changes or goes below the minimum threshold, it is updated and printed.

        Parameters:
        - min_lr_threshold (float): Minimum allowed learning rate. Default is 1e-4.
        """
        # Get the current learning rate
        current_lr = self.get_lr()

        # Check if learning rate has changed or is below the minimum threshold
        if current_lr != self.prev_lr or current_lr < min_lr_threshold:
            # If below the threshold, set the learning rate to the minimum threshold
            if current_lr < min_lr_threshold:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        param.data = param.data.to(dtype=self.dtype)
                    param_group['lr'] = min_lr_threshold
                current_lr = min_lr_threshold  # Update current_lr to reflect the threshold adjustment
                print(f"Learning rate fell below threshold; set to minimum learning rate: {min_lr_threshold:.6f}")
            else:
                print(f"Learning rate changed to: {current_lr:.6f}")

            # Update prev_lr to the new learning rate (either changed or threshold-adjusted)
            self.prev_lr = current_lr


    def process_batch(self, batch):
        """
        Process a batch of data containing both supervised and weakly-supervised data by transferring it 
        to the appropriate device, reshaping it, and applying either float32 or bfloat16 precision based 
        on the training mode.
        """
        try:
            def process_single_mode(data):
                # Use .half() for mixed precision or .float() for single precision
                #dtype = torch.float32 if single_precision else torch.bfloat16

                # Move tensors to the correct device with the appropriate precision
                img = data['img'].to(self.device, dtype=self.dtype)  #[B, 1+N_aug, C, W, H]
                mask = data['mask_semantic'].to(self.device, dtype=self.dtype)  #[B, 1+N_aug, C, W, H]
                labels_affs = data['labels_affs'].to(self.device, dtype=self.dtype)  #[B, 1+N_aug, C, W, H]
                affs_mask = data['affs_mask'].to(self.device, dtype=self.dtype)  #[B, 1+N_aug, C, W, H]
                weight_map = data['weight_map'].to(self.device, dtype=self.dtype)  #[B, 1+N_aug, C, W, H]

                # Reshape dimensions to match the model's expected input format
                return {
                    'img': self.reshape_batch_dim(img),  #[B*(1+N_aug), C, W, H] 
                    'mask': self.reshape_batch_dim(mask),  #[B*(1+N_aug), C, W, H] 
                    'labels_affs': self.reshape_batch_dim(labels_affs), #[B*(1+N_aug), C, W, H] 
                    'affs_mask': self.reshape_batch_dim(affs_mask),  #[B*(1+N_aug), C, W, H] 
                    'weight_map': self.reshape_batch_dim(weight_map), #[B*(1+N_aug), C, W, H] 
                }

            # Process both supervised and weakly-supervised parts of the batch
            supervised_data = process_single_mode(batch['aug_supervised'])  
            weakly_supervised_data = process_single_mode(batch['aug_weakly_supervised'])  

            # Extract batch and augmentation counts
            batch_info = {
                'aug_supervised': {
                    'BatchSize': batch['aug_supervised']['img'].shape[0],
                    'Augmentations': batch['aug_supervised']['img'].shape[1]-1
                },
                'aug_weakly_supervised': {
                    'BatchSize': batch['aug_weakly_supervised']['img'].shape[0],
                    'Augmentations': batch['aug_weakly_supervised']['img'].shape[1]-1
                }
            }
            return {'aug_supervised': supervised_data, 'aug_weakly_supervised': weakly_supervised_data, 'batch_info': batch_info}

        except KeyError as e:
            raise ValueError(f"Key {e} not found in the batch. Expected keys: 'img', 'mask_semantic', 'labels_affs', 'affs_mask', 'weight_map'.")
        except Exception as e:
            raise RuntimeError(f"Error processing batch: {str(e)}")


    def train(self):
        """
        Main training loop, iterates over epochs, performing training, validation, test, and checkpointing.
        """
        print("\nStarting the model training...")

        for epoch in range(self.start_epoch, self.end_epoch):

            # Set models to training or evaluation mode
            self.student_model_handler.model.train()
            self.teacher_model_handler.model.eval()

            # Initialize losses
            epoch_loss_supervised = 0.0
            epoch_loss_weakly_supervised = 0.0

            # Reset cumulative losses for each epoch
            self.cumulative_losses_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'total': 0.0}
            self.cumulative_losses_weakly_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'feature_distillation': 0.0, 'total': 0.0}

            # Reset NaN skip count and gradient norm tracking for each epoch
            self.nan_skip_count = 0
            self.cumulative_grad_norm = 0.0


            # Iterate over training data loader
            successful_batches = 0
            for batch_idx, batch in enumerate(self.train_loader):

                result = self.one_batch_train(batch, batch_idx, epoch)

                # Handle case where batch was skipped due to NaN
                if result is None:
                    continue

                total_loss_supervised, total_loss_weakly_supervised = result

                # Track losses
                epoch_loss_supervised += total_loss_supervised.item()
                epoch_loss_weakly_supervised += total_loss_weakly_supervised.item()
                successful_batches += 1

            # After each epoch, calculate and log average losses, update scheduler, etc.
            num_batches = max(successful_batches, 1)  # Avoid division by zero
            avg_loss_supervised = epoch_loss_supervised / num_batches
            avg_loss_weakly_supervised = epoch_loss_weakly_supervised / num_batches
            avg_grad_norm = self.cumulative_grad_norm / num_batches
            
            # Update scheduler and log losses for both supervised and weakly-supervised data
            self.scheduler.step()
            self.update_learning_rate(min_lr_threshold=1e-4)

            # Log total losses (supervised + weakly-supervised)
            self.writer.add_scalar('Loss/train_supervised', self.cumulative_losses_supervised['total'] / num_batches, epoch)
            self.writer.add_scalar('Loss/train_weakly_supervised', self.cumulative_losses_weakly_supervised['total'] / num_batches, epoch)

            # Log current learning rate
            current_lr = self.get_lr()
            self.writer.add_scalar('LearningRate/train', current_lr, epoch)

            # Log gradient norm and NaN skip count for monitoring stability
            self.writer.add_scalar('Stability/avg_gradient_norm', avg_grad_norm, epoch)
            self.writer.add_scalar('Stability/nan_skip_count', self.nan_skip_count, epoch)
            self.writer.add_scalar('Stability/successful_batches', successful_batches, epoch)

            # Save checkpoints
            self.save_checkpoint(epoch)

            # Validate after each epoch
            avg_val_loss_supervised, avg_val_loss_weakly_supervised, combined_val_loss = self.validate(epoch)

            # Use combined_val_loss for checkpoint saving
            if combined_val_loss < self.best_val_loss:
                self.best_val_loss = combined_val_loss
                self.save_checkpoint(epoch, is_best=True)


            # Print a header row for clarity
            if epoch == self.start_epoch:  # Only print header once, at the beginning
                print(f"{'Epoch':<10}{'Sup Train Loss':<25}{'Weak-sup Train Loss':<25}{'Total Train Loss':<25}{'Sup Val Loss':<25}{'Weak-sup Val Loss':<25}{'Total Val Loss':<25}")

            # Print losses for this epoch
            print(f"{epoch + 1:<10}{avg_loss_supervised:<25.4f}{avg_loss_weakly_supervised:<25.4f}"
                f"{(avg_loss_supervised + self.config['weakly_supervised_weight'] * avg_loss_weakly_supervised):<25.4f}"
                f"{avg_val_loss_supervised:<25.4f}{avg_val_loss_weakly_supervised:<25.4f}"
                f"{combined_val_loss:<25.4f}")

            if self.config['log_gpu_utilization']:
                self.log_gpu_utilization(epoch)
        print("\n ***** Done! ***** \n")


    def one_batch_train(self, batch, batch_idx, epoch):
        # Process the batch for both supervised and weakly-supervised modes
        processed_batch = self.process_batch(batch)

        # Supervised part 
        inp_img_supervised = processed_batch['aug_supervised']['img'] #[B*(1+N_aug_supervised), C, W, H], 1: clean input; N_aug_supervised: N augmenteations
        gt_mask_supervised = processed_batch['aug_supervised']['mask'] #[B*(1+N_aug_supervised), C, W, H] 
        gt_labels_affs_supervised = processed_batch['aug_supervised']['labels_affs'] #[B*(1+N_aug_supervised), C, W, H] 
        labels_affs_mask_supervised = processed_batch['aug_supervised']['affs_mask'] #[B*(1+N_aug_supervised), C, W, H] 
        weight_map_supervised = processed_batch['aug_supervised']['weight_map'] #[B*(1+N_aug_supervised), C, W, H] 
        supervised_batchSize = processed_batch['batch_info']['aug_supervised']['BatchSize'] # B in [B*(1+N_aug_supervised), C, W, H]  
        supervised_augmentationNum = processed_batch['batch_info']['aug_supervised']['Augmentations'] # N_aug_supervised in [B*(1+N_aug_supervised), C, W, H] 
        batch_Mul_clean_augments_supervised = inp_img_supervised.shape[0] # B*(1+N_aug_supervised) in [B*(1+N_aug_supervised), C, W, H] 

        # Weakly supervised part 
        inp_img_weakly_supervised = processed_batch['aug_weakly_supervised']['img'] #[B*(1+N_aug_weaklysupervised), C, W, H]
        gt_mask_weakly_supervised = processed_batch['aug_weakly_supervised']['mask'] #[B*(1+N_aug_weaklysupervised), C, W, H]
        #gt_labels_affs_weakly_supervised = processed_batch['aug_weakly_supervised']['labels_affs'] #[B*(1+N_aug_weaklysupervised), C, W, H]
        #labels_affs_mask_weakly_supervised = processed_batch['aug_weakly_supervised']['affs_mask'] #[B*(1+N_aug_weaklysupervised), C, W, H]
        #weight_map_weakly_supervised = processed_batch['aug_weakly_supervised']['weight_map'] #[B*(1+N_aug_weaklysupervised), C, W, H]
        weakly_supervised_batchSize = processed_batch['batch_info']['aug_weakly_supervised']['BatchSize'] # B in [B*(1+N_aug_weaklysupervised), C, W, H] 
        aug_weakly_supervised_augmentationNum = processed_batch['batch_info']['aug_weakly_supervised']['Augmentations'] # N_aug_weaklysupervised in [B*(1+N_aug_weaklysupervised), C, W, H] 
        
        # Stack the inputs from both supervised and weakly supervised
        inp_img_combined = torch.cat((inp_img_supervised, inp_img_weakly_supervised), dim=0) # [B*(1+N_aug_supervised) + B*(1+N_aug_weaklysupervised), C, W, H]

        # Reset gradients
        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            # Combined weakly supervised data and supervised data is fed to the student modeel
            embedding_combined_student, pred_mask_combined_student = self.student_model_handler.model(inp_img_combined) # inp_img_combined's size [B*(1+N_aug_supervised) + B*(1+N_aug_weaklysupervised), C, W, H]

            # Separate the predictions for supervised and weakly-supervised branches
            #embedding_supervised = embedding_combined_student[:batch_Mul_clean_augments_supervised]
            pred_mask_supervised = pred_mask_combined_student[:batch_Mul_clean_augments_supervised] # [B*(1+N_aug_supervised), C, W, H]

            #embedding_weakly_supervised = embedding_combined_student[batch_Mul_clean_augments_supervised:]
            pred_mask_weakly_supervised = pred_mask_combined_student[batch_Mul_clean_augments_supervised:] # [B*(1+N_aug_weaklysupervised), C, W, H]

            # Weakly supervised image is fed to the teacher model and outputs from teacher are considered as weakly ground truth
            # torch.no_grad() prevents storing intermediate activations for the frozen teacher, saving GPU memory
            with torch.no_grad():
                embedding_weakly_supervised_teacher, pred_mask_weakly_supervised_teacher_WGT = self.teacher_model_handler.model(inp_img_weakly_supervised)

            # Get student embedding for weakly-supervised data (for feature distillation)
            embedding_weakly_supervised_student = embedding_combined_student[batch_Mul_clean_augments_supervised:]

            # Compute supervised and weakly-supervised losses
            total_loss, total_loss_supervised, total_loss_weakly_supervised, pred_affinity_sup = self.compute_losses(
                embedding_combined_student,
                pred_mask_combined_student,
                pred_mask_weakly_supervised_teacher_WGT,
                batch_Mul_clean_augments_supervised,
                gt_labels_affs_supervised,
                weight_map_supervised,
                labels_affs_mask_supervised,
                gt_mask_supervised,
                supervised_batchSize,
                supervised_augmentationNum,
                weakly_supervised_batchSize,
                aug_weakly_supervised_augmentationNum,
                embedding_weakly_supervised_student=embedding_weakly_supervised_student,
                embedding_weakly_supervised_teacher=embedding_weakly_supervised_teacher,
            )

        # Safe backward pass with NaN checking
        backward_success = self.safe_loss_backward(total_loss, batch_idx, epoch)

        if not backward_success:
            # Batch was skipped due to NaN - return None to signal skip
            torch.cuda.empty_cache()
            return None

        # Gradient clipping for stability
        if self.grad_clip_enabled:
            if self.scaler is not None:
                # Unscale gradients before clipping for float16
                self.scaler.unscale_(self.optimizer)

            grad_norm = self.clip_gradients(
                self.student_model_handler.model,
                self.grad_clip_max_norm
            )
            self.cumulative_grad_norm += grad_norm

            # Check for NaN in gradients after clipping
            if not np.isfinite(grad_norm):
                print(f"WARNING: Non-finite gradient norm ({grad_norm}) at epoch {epoch}, batch {batch_idx}")
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                return None

        # Optimizer step
        if self.scaler is not None:
            # float16 path
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # float32 or bfloat16 path
            self.optimizer.step()

        # Update teacher model using EMA
        self.update_teacher_model_ema(self.student_model_handler.model, self.teacher_model_handler.model, self.ema_decay)

        # Update the log_images() call to include the predicted affinities:
        if self.save_condition(epoch, batch_idx):
            self.log_images(
                epoch, batch_idx,
                inp_img_supervised, pred_mask_supervised, gt_mask_supervised, pred_affinity_sup, gt_labels_affs_supervised,
                inp_img_weakly_supervised, pred_mask_weakly_supervised, gt_mask_weakly_supervised,
                mode='train'
            )

        return total_loss_supervised, total_loss_weakly_supervised

    @staticmethod
    def process_weakly_supervised(pred_mask_weakly_supervised_teacher_WGT, 
                                weakly_supervised_batchSize, 
                                aug_weakly_supervised_augmentationNum):
        """
        Extract clean parts from pred_mask_weakly_supervised_teacher_WGT and repeat them across augmentations.

        Args:
            pred_mask_weakly_supervised_teacher_WGT (torch.Tensor): Input tensor of shape [B * (1 + aug_N), C, W, H]
            weakly_supervised_batchSize (int): Batch size B.
            aug_weakly_supervised_augmentationNum (int): Number of augmentations aug_N.

        Returns:
            torch.Tensor: Tensor with clean parts repeated, of shape [B * (1 + aug_N), C, W, H].
        """
        C, W, H = pred_mask_weakly_supervised_teacher_WGT.shape[1:]

        # Step 1: Reshape to [B, 1 + aug_N, C, W, H]
        reshaped = pred_mask_weakly_supervised_teacher_WGT.view(
            weakly_supervised_batchSize, 
            1 + aug_weakly_supervised_augmentationNum, 
            C, W, H
        )

        # Step 2: Extract the clean parts (index 0 in augmentation dimension)
        clean_parts = reshaped[:, 0, :, :, :]  # Shape: [B, C, W, H]

        # Step 3: Repeat clean parts along the augmentation dimension
        repeated_clean_parts = clean_parts.unsqueeze(1).repeat(
            1, 1 + aug_weakly_supervised_augmentationNum, 1, 1, 1
        )  # Shape: [B, 1 + aug_N, C, W, H]

        # Step 4: Reshape back to [B * (1 + aug_N), C, W, H]
        final_clean_parts = repeated_clean_parts.view(-1, C, W, H)

        return final_clean_parts


    def compute_losses(
                        self,
                        embedding_combined_student,
                        pred_mask_combined_student,
                        pred_mask_weakly_supervised_teacher_WGT,
                        batch_Mul_clean_augments_supervised,
                        gt_labels_affs_supervised,
                        weight_map_supervised,
                        labels_affs_mask_supervised,
                        gt_mask_supervised,
                        supervised_batchSize,
                        supervised_augmentationNum,
                        weakly_supervised_batchSize,
                        aug_weakly_supervised_augmentationNum,
                        embedding_weakly_supervised_student=None,
                        embedding_weakly_supervised_teacher=None,
                    ):
        """
        Compute the total loss for both supervised and weakly-supervised data, and return the predicted affinities.
        Includes feature distillation loss between teacher and student embeddings for improved stability.

            Returns:
            total_loss (torch.Tensor): Total loss combining supervised and weakly-supervised losses.
            total_loss_supervised (torch.Tensor): Supervised loss.
            total_loss_weakly_supervised (torch.Tensor): Weakly-supervised loss.
            pred_affinity_sup (torch.Tensor): Predicted affinities for the supervised data.
        """
        # Separate the predictions for supervised and weakly-supervised branches
        embedding_supervised_student = embedding_combined_student[:batch_Mul_clean_augments_supervised,:,:,:]
        pred_mask_supervised_student = pred_mask_combined_student[:batch_Mul_clean_augments_supervised,:,:,:]

        #embedding_weakly_supervised_student = embedding_combined_student[batch_Mul_clean_augments_supervised:,:,:,:]
        pred_mask_weakly_supervised_student = pred_mask_combined_student[batch_Mul_clean_augments_supervised:,:,:,:]


        ### Compute Supervised Losses
        # Embedding loss for supervised data
        embedding_loss_val_sup, pred_affinity_sup, _ = embedding_loss(
                                                                    embedding_supervised_student,
                                                                    gt_labels_affs_supervised,
                                                                    weight_map_supervised,
                                                                    labels_affs_mask_supervised,
                                                                    offsets=self.offsets,
                                                                    criterion="WeightedMSE",
                                                                    affs0_weight=1,
                                                                    mode="AAAI",
                                                                )

        # Mask loss for supervised data
        pred_mask_losses_sup = self.compute_loss_mask(pred_mask_supervised_student, gt_mask_supervised)
        pred_mask_loss_val_sup = pred_mask_losses_sup["total"]


        # Total supervised loss
        total_loss_supervised = (
            self.config["embedding_weight"] * embedding_loss_val_sup
            + self.config["mask_weight"] * pred_mask_loss_val_sup
        )

        ### Compute Weakly-Supervised Losses ###
        """"
        # Mask loss for weakly-supervised data: convert teacher's prediction to class indices
        #weakly_supervised_gt = torch.argmax(pred_mask_weakly_supervised_teacher_WGT, dim=1).long()

        # Take the first instance of the teacher's weakly supervised prediction
        pred_mask_weakly_supervised_teacher_WGT_clean = pred_mask_weakly_supervised_teacher_WGT[:weakly_supervised_batchSize, :, :, :]

        # Apply softmax across classes to get probabilities
        probability_map = torch.softmax(pred_mask_weakly_supervised_teacher_WGT_clean, dim=1)

        # Generate binary mask by selecting the class with the highest probability (argmax)
        #binary_mask_argmax = torch.argmax(probability_map, dim=1).squeeze(1)  # Shape: [B, W, H]

        # Generate binary mask by thresholding foreground class probability
        teacher_threshold_WGT = self.config.get('teacher_threshold_WGT', 0.5)
        binary_teacher_WGT = (probability_map[:, 1, :, :] > teacher_threshold_WGT).unsqueeze(1).int()  # Shape: [B, W, H]

        binary_teacher_WGT_clean_repeat = binary_teacher_WGT.repeat(1+aug_weakly_supervised_augmentationNum, 1, 1, 1)
        """

        #############################################################
        #############################################################

        pred_mask_weakly_supervised_teacher_WGT_clean_repeated = self.process_weakly_supervised(
                                                                                                pred_mask_weakly_supervised_teacher_WGT, 
                                                                                                weakly_supervised_batchSize, 
                                                                                                aug_weakly_supervised_augmentationNum,
                                                                                                )
        
        probability_map = torch.softmax(pred_mask_weakly_supervised_teacher_WGT_clean_repeated, dim=1)

        teacher_threshold_WGT = self.config.get('teacher_threshold_WGT', 0.5)
        binary_teacher_WGT_clean_repeat = (probability_map[:, 1, :, :] > teacher_threshold_WGT).unsqueeze(1).int()  # Shape: [B, W, H]




        #############################################################
        #############################################################






        pred_mask_losses_weak = self.compute_loss_mask(
                                        pred_mask_weakly_supervised_student,
                                        binary_teacher_WGT_clean_repeat,
                                    )

        pred_mask_loss_val_weak = pred_mask_losses_weak["total"]

        # Feature distillation loss: match student embeddings to teacher embeddings
        feature_distillation_loss_val = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if (self.feature_distillation_enabled and
            embedding_weakly_supervised_student is not None and
            embedding_weakly_supervised_teacher is not None):

            # Process teacher embeddings to match student's shape (repeat clean parts)
            embedding_teacher_clean_repeated = self.process_weakly_supervised(
                embedding_weakly_supervised_teacher,
                weakly_supervised_batchSize,
                aug_weakly_supervised_augmentationNum,
            )

            # Compute feature distillation loss (using float32 for stability)
            feature_distillation_loss_val = feature_distillation_loss(
                embedding_weakly_supervised_student.float(),
                embedding_teacher_clean_repeated.float().detach(),  # Detach teacher (no gradients)
                mode='mse',
                reduction='mean'
            )

        # Total weakly-supervised loss (including feature distillation)
        total_loss_weakly_supervised = (
            self.config["mask_weight"] * pred_mask_loss_val_weak +
            self.feature_distillation_weight * feature_distillation_loss_val
        )

        ### Combine Supervised and Weakly-Supervised Losses ###
        # Use float32 for final loss computation to prevent overflow/underflow
        total_loss = (
            total_loss_supervised.float() +
            self.config["weakly_supervised_weight"] * total_loss_weakly_supervised.float()
        )

        # Clamp loss to prevent extreme values
        max_loss_value = 1e6
        if total_loss.item() > max_loss_value:
            print(f"WARNING: Loss clamped from {total_loss.item()} to {max_loss_value}")
            total_loss = torch.clamp(total_loss, max=max_loss_value)

        # Update cumulative losses
        self.cumulative_losses_supervised['embedding_loss'] += embedding_loss_val_sup.item()
        self.cumulative_losses_supervised['mask_loss'] += pred_mask_loss_val_sup.item()
        self.cumulative_losses_supervised['total'] += total_loss_supervised.item()

        self.cumulative_losses_weakly_supervised['feature_distillation'] += feature_distillation_loss_val.item()
        self.cumulative_losses_weakly_supervised['mask_loss'] += pred_mask_loss_val_weak.item()
        self.cumulative_losses_weakly_supervised['total'] += total_loss_weakly_supervised.item()


        return (
            total_loss,
            total_loss_supervised,
            total_loss_weakly_supervised,
            pred_affinity_sup,
        )




    def compute_loss_mask(self, pred_mask, gt_mask):
        """
        Compute and return the weighted loss for the model predictions.
        Uses float32 internally for numerical stability even when model is bfloat16.

        Parameters:
        - pred_mask (Tensor): Predicted logits from the model.
        - gt_mask (Tensor): Ground truth masks.

        Returns:
        - dict: A dictionary containing individual losses and the total loss.
        """
        losses = {'total': torch.tensor(0.0, device=self.device, dtype=torch.float32)}

        # Convert to float32 for stable loss computation
        # pred_mask_f32 = F.relu(pred_mask.float())  # Use float32 for loss computation
        pred_mask_f32 = pred_mask.float()  # Use float32 for loss computation
        gt_mask_f32 = gt_mask.float()

        # Check for NaN in inputs
        if torch.isnan(pred_mask_f32).any() or torch.isnan(gt_mask_f32).any():
            print("WARNING: NaN detected in loss inputs, returning zero loss")
            return losses

        # Compute the loss for each criterion defined in the model handler
        for name, criterion in self.student_model_handler.criterion.items():
            # Use float32 for weight to ensure stable computation
            weight = torch.tensor(self.config['loss_weights'][name], device=self.device, dtype=torch.float32)

            try:
                loss = weight * criterion(pred_mask_f32, gt_mask_f32.squeeze(1).long())

                # Check for NaN/Inf in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN/Inf detected in {name} loss, skipping")
                    continue

                losses[name] = loss
                losses['total'] = losses['total'] + loss

            except RuntimeError as e:
                print(f"WARNING: Error computing {name} loss: {e}")
                continue

        return losses


    def validate(self, epoch):
        """
        Perform validation and return the average losses for supervised and weakly-supervised data.
        Additionally, log images from both the student and teacher models.
        """
        # Set models to evaluation mode
        self.student_model_handler.model.eval()
        self.teacher_model_handler.model.eval()
        
        # Initialize losses for both supervised and weakly-supervised data
        val_loss_supervised = 0.0
        val_loss_weakly_supervised = 0.0

        self.cumulative_losses_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'total': 0.0}
        self.cumulative_losses_weakly_supervised = {'embedding_loss': 0.0, 'mask_loss': 0.0, 'feature_distillation': 0.0, 'total': 0.0}

        # Disable gradients since we're only validating
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Process the batch
                processed_batch = self.process_batch(batch)

                # Supervised part 
                inp_img_supervised = processed_batch['aug_supervised']['img'] #[B*(1+N_aug_supervised), C, W, H] 
                gt_mask_supervised = processed_batch['aug_supervised']['mask'] #[B*(1+N_aug_supervised), C, W, H] 
                gt_labels_affs_supervised = processed_batch['aug_supervised']['labels_affs'] #[B*(1+N_aug_supervised), C, W, H] 
                labels_affs_mask_supervised = processed_batch['aug_supervised']['affs_mask'] #[B*(1+N_aug_supervised), C, W, H] 
                weight_map_supervised = processed_batch['aug_supervised']['weight_map'] #[B*(1+N_aug_supervised), C, W, H] 
                supervised_batchSize = processed_batch['batch_info']['aug_supervised']['BatchSize'] # B in [B*(1+N_aug_supervised), C, W, H]  
                supervised_augmentationNum = processed_batch['batch_info']['aug_supervised']['Augmentations'] # N_aug_supervised in [B*(1+N_aug_supervised), C, W, H] 
                batch_Mul_clean_augments_supervised = inp_img_supervised.shape[0] # B*(1+N_aug_supervised) in [B*(1+N_aug_supervised), C, W, H] 

                # Weakly supervised part 
                inp_img_weakly_supervised = processed_batch['aug_weakly_supervised']['img'] #[B*(1+N_aug_weaklysupervised), C, W, H]
                gt_mask_weakly_supervised = processed_batch['aug_weakly_supervised']['mask'] #[B*(1+N_aug_weaklysupervised), C, W, H]
                weakly_supervised_batchSize = processed_batch['batch_info']['aug_weakly_supervised']['BatchSize'] # B in [B*(1+N_aug_weaklysupervised), C, W, H] 
                aug_weakly_supervised_augmentationNum = processed_batch['batch_info']['aug_weakly_supervised']['Augmentations'] # N_aug in [B*(1+N_aug_weaklysupervised), C, W, H] 
                
                # Stack the inputs from both supervised and weakly supervised
                inp_img_combined = torch.cat((inp_img_supervised, inp_img_weakly_supervised), dim=0) # [B*(1+N_aug_supervised) + B*(1+N_aug_weaklysupervised), C, W, H]

                # Begin mixed-precision context with autocast
                with torch.amp.autocast(self.device.type, dtype=self.dtype):

                    # Forward pass on the student model
                    embedding_combined_student, pred_mask_combined_student = self.student_model_handler.model(inp_img_combined)

                    # Forward pass on the teacher model for weakly-supervised data
                    embedding_weakly_supervised_teacher, pred_mask_weakly_supervised_teacher_WGT = self.teacher_model_handler.model(inp_img_weakly_supervised)

                    # Get student embedding for weakly-supervised data (for feature distillation)
                    embedding_weakly_supervised_student = embedding_combined_student[batch_Mul_clean_augments_supervised:]

                    # Compute losses
                    total_loss, total_loss_supervised, total_loss_weakly_supervised, pred_affinity_sup = self.compute_losses(
                        embedding_combined_student,
                        pred_mask_combined_student,
                        pred_mask_weakly_supervised_teacher_WGT,
                        batch_Mul_clean_augments_supervised,
                        gt_labels_affs_supervised,
                        weight_map_supervised,
                        labels_affs_mask_supervised,
                        gt_mask_supervised,
                        supervised_batchSize,
                        supervised_augmentationNum,
                        weakly_supervised_batchSize,
                        aug_weakly_supervised_augmentationNum,
                        embedding_weakly_supervised_student=embedding_weakly_supervised_student,
                        embedding_weakly_supervised_teacher=embedding_weakly_supervised_teacher,
                    )

                # Accumulate total losses
                val_loss_supervised += total_loss_supervised.item()
                val_loss_weakly_supervised += total_loss_weakly_supervised.item()

                # Log images if the save condition is met (e.g., specific epoch and batch)
                if self.save_condition(epoch, batch_idx):
                    # Log predictions from the student model
                    self.log_images(
                        epoch, batch_idx,
                        inp_img_supervised, pred_mask_combined_student[:batch_Mul_clean_augments_supervised], gt_mask_supervised, pred_affinity_sup, gt_labels_affs_supervised,
                        inp_img_weakly_supervised, pred_mask_combined_student[batch_Mul_clean_augments_supervised:], gt_mask_weakly_supervised,
                        mode='val_student'
                    )

                    # Log teacher's predictions for weakly-supervised data
                    self.log_images(
                        epoch, batch_idx,
                        inp_img_supervised, pred_mask_combined_student[:batch_Mul_clean_augments_supervised], gt_mask_supervised, pred_affinity_sup, gt_labels_affs_supervised,
                        inp_img_weakly_supervised, pred_mask_weakly_supervised_teacher_WGT, gt_mask_weakly_supervised,
                        mode='val_teacher'  # Different mode to distinguish teacher outputs
                    )

        # Calculate average losses
        avg_val_loss_supervised = val_loss_supervised / len(self.val_loader)
        avg_val_loss_weakly_supervised = val_loss_weakly_supervised / len(self.val_loader)
        combined_val_loss = avg_val_loss_supervised + self.config['weakly_supervised_weight'] * avg_val_loss_weakly_supervised

        # Log total losses to TensorBoard
        self.writer.add_scalar('Loss/val_supervised', self.cumulative_losses_supervised['total'] / len(self.val_loader), epoch)
        self.writer.add_scalar('Loss/val_weakly_supervised', self.cumulative_losses_weakly_supervised['total'] / len(self.val_loader), epoch)

        # Log losses to a text file
        with open(os.path.join(self.config['log_dir'], 'metrics_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch}: Validation Losses\n')
            f.write(f'  Supervised Loss: {avg_val_loss_supervised}\n')
            f.write(f'  Weakly-supervised Loss: {avg_val_loss_weakly_supervised}\n')
            f.write(f'  Combined Loss: {combined_val_loss}\n\n')

        return avg_val_loss_supervised, avg_val_loss_weakly_supervised, combined_val_loss



    def save_condition(self, epoch, batch_idx):
        """
        Determine whether to save images based on the current epoch and batch index.

        Parameters:
        - epoch (int): The current epoch number.
        - batch_idx (int): The current batch index.

        Returns:
        - bool: True if images should be saved, False otherwise.
        """
        return (self.config['save_image_frequency'] and 
                (epoch + 1) % self.config['save_image_frequency'] == 0 and 
                batch_idx == 0)

   
    #@staticmethod
    def one_hot_from_single_channel(self, single_channel: torch.Tensor, num_classes: int) -> torch.Tensor:
        # Convert single_channel to int data type
        single_channel = single_channel.long() 

        # Apply one-hot encoding
        one_hot_single_channel = F.one_hot(single_channel.squeeze(1), num_classes)
        one_hot_single_channel = one_hot_single_channel.permute(0, 3, 1, 2)
       
        return one_hot_single_channel.to(self.device, dtype=self.dtype) 


    def log_gpu_utilization(self, epoch):
        """
        Logs the GPU utilization and memory information.

        Parameters:
        - epoch (int): The current epoch number.
        """
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Log GPU utilization
            self.writer.add_scalar(f'GPU_{i}/Utilization', util.gpu, epoch)
            self.writer.add_scalar(f'GPU_{i}/Memory_Utilization', util.memory, epoch)
            self.writer.add_scalar(f'GPU_{i}/Memory_Used_MiB', mem_info.used / 1024**2, epoch)
            self.writer.add_scalar(f'GPU_{i}/Memory_Total_MiB', mem_info.total / 1024**2, epoch)
        
        # Shutdown NVML
        pynvml.nvmlShutdown()

