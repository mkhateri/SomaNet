import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import pynvml
from torch import Tensor
import cv2

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
from model.losses.losses import embedding_loss 

import torch.nn.functional as F
from utils.affinity import multi_offset
import seaborn as sns  # Import seaborn for density plots


class Trainer:
    """Trainer class to handle the training, validation, and testing of a model"""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.config = ConfigHandler(config).config
        set_seed(self.config.get('random_seed', 321)) 
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = min(torch.cuda.device_count(), self.config.get('num_gpus', 1))
        
        if self.num_gpus > 1:
            if self.config['batch_size'] % self.num_gpus != 0:
                raise ValueError("Batch size must be divisible by the number of GPUs.")
        print(f"Using {self.num_gpus} GPUs for training.")

        self.dtype = self.map_precision(self.config.get('data_precision', 'float32'))

        # Single/Mixed precision training
        # self.single_precision = self.config.get('single_precision', True)
        # self.dtype = torch.float32 if self.single_precision else torch.bfloat16 

        # Data loaders
        self.train_loader, self.val_loader = DataHandler(self.config)._get_train_dataloaders()
        
        # Model handler
        self.model_handler = ModelHandler(self.config, self.device)

        # convert_model_dtype
        #self.model_handler.model = convert_model_dtype(self.model_handler.model, dtype=self.dtype)

        # Apply parallelization if necessary
        self.model_handler.model = ParallelHandler.apply_parallel(self.model_handler.model, self.num_gpus)
        
        # Move the model to the appropriate device
        self.model_handler.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = self.model_handler.optimizer
        self.scheduler = self.model_handler.scheduler   # CombinedScheduler class will be added including several scheduler:warmup and refining 

        # Initialize padding_size
        self.padding_size = self.config.get('padding_size', 0)

        # Initialize offsets
        self.offsets = multi_offset(shifts=self.config['shifts_affinity'], neighbor=self.config['neighbor_affinity'])

        # Training state
        self.best_val_loss = float('inf')
        self.prev_lr = None
        self.start_epoch = 0  
        self.end_epoch = self.config['epochs']

        # Single/Mixed precision training
        # self.scaler = torch.amp.GradScaler(device=self.device.type) if self.dtype != torch.float32 else None
        self.scaler = torch.cuda.amp.GradScaler() if self.dtype != torch.float32 else None        
        
        # Resume training or fine-tuning or training from scratch
        if self.config['resume']:
            checkpoint_type = self.config.get('checkpoint_type', 'latest')
            checkpoint_path = self.get_checkpoint_path(checkpoint_type)
            self.load_checkpoint(checkpoint_path)
            self.prev_lr = self.get_lr()
            print(f"Resuming training from epoch {self.start_epoch + 1} with learning rate: {self.prev_lr}")
        # elif self.config['fine_tune']:
        #     self.load_pretrained_weights(self.config['pretrained_model_dir'])
        #     self.start_epoch = 0
        #     # TensorBoard writer
        #     start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        #     log_dir = os.path.join(self.config.get('log_dir', './logs'), start_time)
        #     self.writer = SummaryWriter(log_dir=log_dir)
        else:
            # TensorBoard writer
            start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join(self.config.get('log_dir', './logs'), start_time)
            self.writer = SummaryWriter(log_dir=log_dir)

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
            'state_dict': self.model_handler.model.module.state_dict() if self.num_gpus > 1 else self.model_handler.model.state_dict(),
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
        state_dict = ParallelHandler.adjust_state_dict(checkpoint['state_dict'], self.num_gpus)
        self.model_handler.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.end_epoch = self.start_epoch + self.config['epochs']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.writer = SummaryWriter(log_dir=checkpoint['log_dir'])
        print(f"Loaded checkpoint from epoch {self.start_epoch}")

    def load_pretrained_weights(self, pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
        state_dict = ParallelHandler.adjust_state_dict(checkpoint['state_dict'], self.num_gpus)
        self.model_handler.model.load_state_dict(state_dict)
        print(f"Loaded pre-trained model from {pretrained_model_path}")

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']


    # Modify the function to include thresholded segmentation maps
    def log_images(self, epoch, batch_idx, inp_img, pred_mask, gt_mask, pred_affinity, gt_labels_affs, mode='train'):
        """
        Logs images to the specified directory. The logged images include the input, 
        probability maps for each class, segmentation map, and ground truth.

        Args:
            epoch (int): The current epoch number.
            batch_idx (int): The current batch index.
            inp_img (torch.Tensor): The input images.
            pred_mask (torch.Tensor): The predicted logits from the model.
            gt_mask (torch.Tensor): The ground truth segmentation masks.
            pred_affinity (torch.Tensor): The predicted affinity map.
            gt_labels_affs (torch.Tensor): The ground truth affinity map.
            mode (str): The mode in which the model is operating (e.g., 'train', 'val').
        """
        # Inner function to remove small objects and apply morphology if necessary
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
        
        # Define the directory to save images
        save_dir = os.path.join(self.config['log_dir'], mode, f'{mode}_results_epoch_{epoch + 1}_batch_{batch_idx}')
        os.makedirs(save_dir, exist_ok=True)

        # Convert logits to probabilities
        prob_map = F.softmax(pred_mask, dim=1)  # Use softmax to get probabilities for each class

        # Detach tensors and convert to numpy arrays
        inp_img_np = inp_img[0, 0].to(torch.float32).detach().cpu().numpy()  # Extract first batch and first channel
        prob_map_np = prob_map[0].to(torch.float32).detach().cpu().numpy()   # Extract first batch
        gt_mask_np = gt_mask[0, 0].to(torch.float32).detach().cpu().numpy()  # Extract first batch and first channel
        pred_affinity_np = pred_affinity[0].to(torch.float32).detach().cpu().numpy()  # Extract first batch
        gt_affinity_np = gt_labels_affs[0].to(torch.float32).detach().cpu().numpy()   # Extract first batch

        # Compute argmax segmentation map
        argmax_segmentation_map = np.argmax(pred_mask[0].to(torch.float32).detach().cpu().numpy(), axis=0).astype(np.uint8)
        argmax_segmentation_map = process_map(argmax_segmentation_map)  # Apply post-processing

        # Apply thresholding for visualization
        thresholds = [0.25, 0.35, 0.4, 0.45]
        threshold_maps = [(prob_map_np[1] > t).astype(np.uint8) for t in thresholds]



        threshold_maps_cleaned = [process_map(thresh_map) for thresh_map in threshold_maps]
        
        # Create a figure with 3 rows and 4 columns for images
        fig, axes = plt.subplots(3, 4, figsize=(28, 15))
        titles = ['Input Image', 'Probability Map (Class 0)', 
                'Probability Map (Class 1)', 'Ground Truth Mask',
                'Class 0 Histogram', 'Class 1 Histogram',
                'Argmax Segmentation', 'Ground Truth Affinity'] + \
                [f'Threshold {t}' for t in thresholds]  # Titles for threshold maps

        # Display images
        images = [
            inp_img_np,                   # Input Image
            prob_map_np[0],               # Probability map for Class 1
            prob_map_np[1],    # Cleaned segmentation map at threshold 0.2
            gt_mask_np,                   # Ground Truth Mask
            None,                         # Histogram for Class 0 Probability Map (placeholder)
            None,                         # Histogram for Class 1 Probability Map (placeholder)
            argmax_segmentation_map,      # Argmax Segmentation Map
            gt_affinity_np[0]             # Ground Truth Affinity
        ] + threshold_maps_cleaned  # Append the cleaned maps at different thresholds

        # Plot histograms for Class 0 and Class 1 probabilities in the second row
        for idx, (ax, title) in enumerate(zip(axes.flatten(), titles)):
            ax.set_title(title)
            ax.axis('off')

            if idx == 4:  # Class 0 histogram
                min_val, max_val = prob_map_np[0].min(), prob_map_np[0].max()
                hist_data, bin_edges, _ = ax.hist(prob_map_np[0].ravel(), bins=50, range=(min_val, max_val), 
                                                color='blue', edgecolor='black', alpha=0.7)
                mean_val, std_val = prob_map_np[0].mean(), prob_map_np[0].std()
                ax.text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}",
                        transform=ax.transAxes, color='black', fontsize=10,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                # Set x-ticks as bin edges with a small rotation
                ax.set_xticks(bin_edges[::5])  # Showing every 5th bin edge to avoid clutter
                ax.set_xticklabels([f"{x:.2f}" for x in bin_edges[::5]], rotation=45, ha='right')
                ax.set_xlabel('Probability Value')
                ax.set_ylabel('Frequency')

            elif idx == 5:  # Class 1 histogram
                min_val, max_val = prob_map_np[1].min(), prob_map_np[1].max()
                hist_data, bin_edges, _ = ax.hist(prob_map_np[1].ravel(), bins=50, range=(min_val, max_val), 
                                                color='orange', edgecolor='black', alpha=0.7)
                mean_val, std_val = prob_map_np[1].mean(), prob_map_np[1].std()
                ax.text(0.05, 0.95, f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}",
                        transform=ax.transAxes, color='black', fontsize=10,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
                # Set x-ticks as bin edges with a small rotation
                ax.set_xticks(bin_edges[::5])  # Showing every 5th bin edge to avoid clutter
                ax.set_xticklabels([f"{x:.2f}" for x in bin_edges[::5]], rotation=45, ha='right')
                ax.set_xlabel('Probability Value')
                ax.set_ylabel('Frequency')

            elif idx == 1:  # Probability Map for Class 1 (add min/max values)
                ax.imshow(prob_map_np[1], cmap='gray')
                min_val, max_val = prob_map_np[1].min(), prob_map_np[1].max()
                ax.text(0.05, 0.95, f"Min: {min_val:.4f}\nMax: {max_val:.4f}",
                        transform=ax.transAxes, color='white', fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='black', alpha=0.7))

            elif idx == 2:  # Probability Map for Class 0 (add min/max values)
                ax.imshow(prob_map_np[0], cmap='gray')
                min_val, max_val = prob_map_np[0].min(), prob_map_np[0].max()
                ax.text(0.05, 0.95, f"Min: {min_val:.4f}\nMax: {max_val:.4f}",
                        transform=ax.transAxes, color='white', fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='black', alpha=0.7))

            elif images[idx] is not None:
                ax.imshow(images[idx], cmap='gray')

        # Adjust layout and save figure
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'results_with_thresholds.png'))
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
    def one_hot_to_single_channel(one_hot: Tensor) -> Tensor:
        """
        Convert a one-hot encoded tensor to single-channel format by taking the argmax along the channel dimension.

        Parameters:
        - one_hot (Tensor): One-hot encoded tensor. Shape: [batch_size, num_classes, height, width]

        Returns:
        - Tensor: Single-channel tensor. Shape: [batch_size, 1, height, width]
        """
        return torch.argmax(one_hot, dim=1, keepdim=True)


    def train(self):
        print("Starting the model training...")
        for epoch in range(self.start_epoch, self.end_epoch):
            self.model_handler.model.train()
            epoch_loss = 0.0
            cumulative_losses = {'embedding_loss': 0.0, 'pred_mask_loss': 0.0, 'total': 0.0}
            
            for batch_idx, batch in enumerate(self.train_loader):
                inp_img = batch['img'].to(self.device, dtype=self.dtype)
                #gt_mask = batch['mask'].to(self.device, dtype=self.dtype)
                gt_mask = batch['mask_semantic'].to(self.device, dtype=self.dtype)
                gt_labels_affs = batch['labels_affs'].to(self.device, dtype=self.dtype)
                labels_affs_mask = batch['affs_mask'].to(self.device, dtype=self.dtype)
                weight_map = batch['weight_map'].to(self.device, dtype=self.dtype)

                self.optimizer.zero_grad()
                # with torch.amp.autocast(self.device.type, dtype=self.dtype):
                with torch.cuda.amp.autocast():
                    embedding, pred_mask = self.model_handler.model(inp_img)

                    # Compute embedding loss
                    embedding_loss_val, pred_affinity, _ = embedding_loss(
                        embedding, gt_labels_affs, weight_map, labels_affs_mask,
                        offsets=self.offsets, criterion='WeightedMSE', affs0_weight=1, mode='AAAI'
                    )
                                                                    
                    pred_mask_losses = self.compute_loss(pred_mask, gt_mask)
                    pred_mask_loss_val = pred_mask_losses['total']
                    
                    # Combine both losses
                    total_loss = self.config['embedding_weight']*embedding_loss_val + self.config['mask_weight']*pred_mask_loss_val

                    #print(f"embedding_loss_val:{embedding_loss_val}, embedding_loss_val: {pred_mask_loss_val}", flush=True)
                
                if self.dtype == torch.float32:
                    # Backpropagation with full precision (FP32)
                    total_loss.backward()
                    # Update weights
                    self.optimizer.step()
                else:
                    # Backpropagation with mixed precision (FP16)
                    self.scaler.scale(total_loss).backward()
                    # Update weights using the scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                epoch_loss += total_loss.item()
                cumulative_losses['embedding_loss'] += embedding_loss_val.item()
                cumulative_losses['pred_mask_loss'] += pred_mask_loss_val.item()
                cumulative_losses['total'] += total_loss.item()
                    
                if self.save_condition(epoch, batch_idx):
                    self.log_images(epoch, batch_idx, inp_img, pred_mask, gt_mask, pred_affinity, gt_labels_affs, mode='train')
                    #self.log_images(epoch, batch_idx, inp_img, pred_mask, gt_mask, mode='train')

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_cumulative_losses = {name: value / len(self.train_loader) for name, value in cumulative_losses.items()}
            
            self.scheduler.step()

            self.writer.add_scalar('Loss/train', avg_cumulative_losses['total'], epoch)

            current_lr = self.get_lr()
            if self.prev_lr is None or current_lr != self.prev_lr:
                self.prev_lr = current_lr
                print(f"Epoch {epoch + 1}/{self.end_epoch} - Learning Rate: {current_lr}")

            self.writer.add_scalar('LearningRate/train', current_lr, epoch)

            self.save_checkpoint(epoch)

            if self.config['save_checkpoint_frequency'] and (epoch + 1) % self.config['save_checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, is_specific=True)
            
            val_loss = self.validate(epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            print(f"Epoch {epoch + 1}/{self.end_epoch} completed with training loss: {avg_epoch_loss:.4f}, validation loss: {val_loss:.4f}")
            if self.config['log_gpu_utilization']:
                self.log_gpu_utilization(epoch)
        print("\n ***** Done! ***** \n")



    def validate(self, epoch):
        self.model_handler.model.eval()
        val_loss = 0.0
        cumulative_losses = {
            'pred_mask_loss': 0.0,
            'embedding_loss': 0.0,
            'total': 0.0
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                inp_img = batch['img'].to(self.device, dtype=self.dtype)
                gt_mask = batch['mask_semantic'].to(self.device, dtype=self.dtype)
                gt_labels_affs = batch['labels_affs'].to(self.device, dtype=self.dtype)
                labels_affs_mask = batch['affs_mask'].to(self.device, dtype=self.dtype)
                weight_map = batch['weight_map'].to(self.device, dtype=self.dtype)

                # Use AMP for mixed precision
                # with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                with torch.cuda.amp.autocast():
                    embedding, pred_mask = self.model_handler.model(inp_img)

                    # Compute embedding loss
                    embedding_loss_val, pred_affinity, _ = embedding_loss(
                        embedding, gt_labels_affs, weight_map, labels_affs_mask,
                        offsets=self.offsets, criterion='WeightedMSE', affs0_weight=1, mode='AAAI'
                    )

                    pred_mask_losses = self.compute_loss(pred_mask, gt_mask)
                    pred_mask_loss_val = pred_mask_losses['total']

                    # Combine losses
                    total_loss = (
                        self.config['embedding_weight'] * embedding_loss_val +
                        self.config['mask_weight'] * pred_mask_loss_val
                    )

                val_loss += total_loss.item()
                cumulative_losses['embedding_loss'] += embedding_loss_val.item()
                cumulative_losses['pred_mask_loss'] += pred_mask_loss_val.item()
                cumulative_losses['total'] += total_loss.item()

                if self.save_condition(epoch, batch_idx):
                    self.log_images(epoch, batch_idx, inp_img, pred_mask, gt_mask, pred_affinity, gt_labels_affs, mode='val')

        avg_val_loss = val_loss / len(self.val_loader)
        avg_cumulative_losses = {name: value / len(self.val_loader) for name, value in cumulative_losses.items()}

        self.writer.add_scalar('Loss/val', avg_cumulative_losses['total'], epoch)

        return avg_val_loss


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


    def compute_loss(self, pred_mask, gt_mask):
        """
        Compute and return the weighted loss for the model predictions.

        Parameters:
        - pred_mask (Tensor): Predicted logits from the model.
        - gt_mask (Tensor): Ground truth masks.

        Returns:
        - dict: A dictionary containing individual losses and the total loss.
        """
        losses = {'total': 0.0}
        #total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float)  # Initialize total_loss as a float tensor
        num_classes = self.config['num_class']

        # One-hot encoding converts [batchsize, 1, width, height] to [batchsize, num_classes, width, height]
        #one_hot_gt_mask = self.one_hot_from_single_channel(gt_mask, num_classes)

        # Compute the loss for each criterion defined in the model handler
        for name, criterion in self.model_handler.criterion.items():
            weight = self.config['loss_weights'][name]
            weight_tensor = torch.tensor(weight, device=self.device, dtype=pred_mask.dtype)  # Ensure weight_tensor is the same type as pred_mask

            loss = weight_tensor * criterion(pred_mask, gt_mask.squeeze(1).long())
            losses[name] = loss  # Store the loss tensor itself
            losses['total'] += loss  # Accumulate total_loss as a tensor

        return losses

    
    #@staticmethod
    def one_hot_from_single_channel(self, single_channel: torch.Tensor, num_classes: int) -> torch.Tensor:
        # Convert single_channel to int data type
        single_channel = single_channel.long() 

        # Apply one-hot encoding
        one_hot_single_channel = F.one_hot(single_channel.squeeze(1), num_classes)
        one_hot_single_channel = one_hot_single_channel.permute(0, 3, 1, 2)
       
        return one_hot_single_channel.to(dtype=self.dtype)




    @staticmethod
    def remove_padding(tensor: torch.Tensor, padding_size: int) -> torch.Tensor:
        """
        Remove padding of size padding_size from all sides of the last two dimensions of a tensor.

        Parameters:
        - tensor (torch.Tensor): The input tensor with padding.
        - padding_size (int): The size of padding to remove from each side.

        Returns:
        - torch.Tensor: The tensor with padding removed from the last two dimensions.
        """
        # Ensure the tensor has at least 2 dimensions
        if tensor.dim() < 2:
            raise ValueError("Input tensor must have at least 2 dimensions.")

        # Use slicing to remove the padding from the last two dimensions
        return tensor[..., padding_size:-padding_size, padding_size:-padding_size]



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


