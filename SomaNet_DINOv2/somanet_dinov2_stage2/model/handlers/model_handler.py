import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Tuple, Union
import torch.optim as optim
import importlib
from utils.utils import set_seed

from model.architectures import ARCHITECTURE_REGISTRY, DINOv2Seg
from model.losses import LOSS_REGISTRY

class ModelHandler:
    def __init__(self, config: Dict[str, Any], device: torch.device, is_teacher: bool):
        self.config = config
        set_seed(self.config.get('random_seed', 321))

        self.device = device
        self.model = self._get_model(model_name=config['model'],
                                     in_channels=config['in_channels'],
                                     num_class=config['num_class'],
                                     image_size=config['crop_size'],
                                     )

        self.criterion = self._get_criterion(losses_name=config['losses'])

        # Freeze DINOv2 backbone (all but last n_unfrozen_blocks ViT blocks) before building optimizer,
        # so the optimizer only tracks parameters that will actually receive gradients.
        self._apply_backbone_freeze()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_model(self, model_name: str, in_channels: int, num_class: int, image_size: int) -> nn.Module:
        Model = ARCHITECTURE_REGISTRY.get(model_name)
        if Model is None:
            raise ValueError(f"Model architecture '{model_name}' is not defined in ARCHITECTURE_REGISTRY")
        use_checkpoint = self.config.get('use_checkpoint', True)
        if model_name == 'DINOv2Seg':
            return Model(
                model_name=self.config.get('dinov2_variant', 'dinov2_vits14'),
                in_channels=in_channels,
                num_classes=num_class,
                emb_dim=self.config.get('emb_dim', 32),
                embed_dim=self.config.get('embed_dim', 64),
                decoder_mid_dim=self.config.get('decoder_mid_dim', 128),
                adapter_depth=self.config.get('adapter_depth', 4),
                use_checkpoint=use_checkpoint,
                pretrained=self.config.get('pretrained', True),
                img_size=image_size,
            ).to(self.device)
        return Model(img_size=image_size, in_chans=in_channels, num_classes=num_class, use_checkpoint=use_checkpoint).to(self.device)

    def _apply_backbone_freeze(self):
        base = self.model.module if hasattr(self.model, 'module') else self.model
        if isinstance(base, DINOv2Seg):
            n_unfrozen = self.config.get('n_unfrozen_blocks', 0)
            base.freeze_backbone(n_unfrozen=n_unfrozen)
    
    def _get_criterion(self, losses_name: list) -> Dict[str, nn.Module]:
        """Initialize criterion (loss functions) as per LOSS_REGISTRY."""
        Losses_dict = {}
        for loss_name in losses_name:
            if loss_name in LOSS_REGISTRY:
                LossClass = LOSS_REGISTRY[loss_name]
                Losses_dict[loss_name] = LossClass().to(self.device)
            else:
                raise ValueError(f"Loss '{loss_name}' is not defined in LOSS_REGISTRY")
        return Losses_dict
    
    def _get_optimizer(self) -> optim.Optimizer:
        optimizer_type = self.config.get('optimizer_name', 'adam')  # Default to 'adam'
        if optimizer_type is None:
            raise KeyError("Optimizer type ('optimizer_name') must be specified in the configuration.")

        optimizer_type = optimizer_type.lower()
        lr = self.config.get('lr_base', 1e-4)  # Default learning rate

        # DINOv2Seg: differential LR for backbone vs new layers (matches pretraining setup)
        base = self.model.module if hasattr(self.model, 'module') else self.model
        if isinstance(base, DINOv2Seg):
            lr_backbone = self.config.get('lr_backbone', lr / 50.0)
            params = base.get_param_groups(lr_backbone=lr_backbone, lr_new=lr)
            print(f"[ModelHandler] DINOv2Seg optimizer: backbone_lr={lr_backbone:.1e}, new_lr={lr:.1e}")
        else:
            params = self.model.parameters()

        if optimizer_type == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer_type == 'adam':
            return optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=0.01, weight_decay=1e-6, amsgrad=True)
        elif optimizer_type == 'adamw':
            return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_type}' is not implemented. Please choose 'sgd', 'adam', or 'adamw'.")

    def _get_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        lr_mode = self.config.get('lr_mode', 'steplr').lower()  # Default to 'steplr'
        total_iters = self.config.get('total_iters', 5000)
        step_size = self.config.get('step_size', 30)
        gamma = self.config.get('gamma', 0.1)
        milestones = self.config.get('milestones', [100000, 150000])
        patience = self.config.get('patience', 10)
        factor = self.config.get('factor', 0.1)
        eta_min = self.config.get('eta_min', 0)  # Default eta_min for CosineAnnealingLR
        T_max = self.config.get('cosine_t_max', 50)  # Default T_max for CosineAnnealingLR
        max_lr = self.config.get('onecycle_max_lr', 1e-3)  # Default max_lr for OneCycleLR
        div_factor = self.config.get('onecycle_div_factor', 25.0)  # Default div_factor for OneCycleLR
        final_div_factor = self.config.get('onecycle_final_div_factor', 10000.0)  # Default final_div_factor for OneCycleLR

        if lr_mode == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif lr_mode == 'multi_steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        elif lr_mode == 'explr':
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif lr_mode == 'lambdalr':
            lambda_func = lambda epoch: (1.0 - epoch / total_iters) ** 0.9
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_func)
        elif lr_mode == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)
        elif lr_mode == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif lr_mode == 'onecyclelr':
            steps_per_epoch = self.config.get('steps_per_epoch', len(self.train_loader))
            scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=self.config['epochs'], div_factor=div_factor, final_div_factor=final_div_factor)
        else:
            raise ValueError(f"Unsupported LR mode: {lr_mode}")

        return scheduler
