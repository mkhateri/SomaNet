import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Tuple, Union
import torch.optim as optim
import importlib
from utils.utils import set_seed

from model.architectures import ARCHITECTURE_REGISTRY
from model.losses import LOSS_REGISTRY

class ModelHandler:
    """
    Handler for model instantiation, loss functions, metrics, optimizer, and scheduler.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        set_seed(self.config.get('random_seed', 321))

        self.device = device
        self.model = self._get_model()

        self.criterion = self._get_criterion(losses_name=config['losses'])
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_model(self) -> nn.Module:
        """Instantiate the model from ARCHITECTURE_REGISTRY (config['model'])."""
        config = self.config

        model_name = config.get('model', 'SwinIR')
        in_channels = config.get('in_channels', config.get('in_chans', 1))
        num_classes = config.get('num_class', config.get('num_classes', 2))
        image_size = config.get('crop_size', config.get('img_size', 512))

        # Check if model is in registry
        if model_name in ARCHITECTURE_REGISTRY:
            model_or_factory = ARCHITECTURE_REGISTRY[model_name]

            # If it's a class (like SwinIR), instantiate with original parameters
            if isinstance(model_or_factory, type):
                if model_name == 'SwinIR':
                    return model_or_factory(
                        img_size=image_size,
                        in_chans=in_channels,
                        num_classes=num_classes
                    ).to(self.device)
                else:
                    return model_or_factory(
                        in_channels=in_channels,
                        num_classes=num_classes
                    ).to(self.device)

            # If it's a factory function (lambda), call it with new-style parameters
            else:
                return model_or_factory(
                    in_channels=in_channels,
                    num_classes=num_classes,
                    mode=config.get('encoder_mode', 'frozen'),
                ).to(self.device)

        raise ValueError(
            f"Model '{model_name}' not found in ARCHITECTURE_REGISTRY. "
            f"Available models: {list(ARCHITECTURE_REGISTRY.keys())}"
        )

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
        """Initialize optimizer based on config."""
        optimizer_type = self.config.get('optimizer_name', 'adam')
        if optimizer_type is None:
            raise KeyError("Optimizer type ('optimizer_name') must be specified in the configuration.")

        optimizer_type = optimizer_type.lower()
        lr = self.config.get('lr_base', 1e-4)

        if optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=0.01, weight_decay=1e-6, amsgrad=True)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_type}' is not implemented. Please choose 'sgd', 'adam', or 'adamw'.")

    def _get_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Initialize learning rate scheduler based on config."""
        lr_mode = self.config.get('lr_mode', 'steplr').lower()
        total_iters = self.config.get('total_iters', 5000)
        step_size = self.config.get('step_size', 30)
        gamma = self.config.get('gamma', 0.1)
        milestones = self.config.get('milestones', [100000, 150000])
        patience = self.config.get('patience', 10)
        factor = self.config.get('factor', 0.1)
        eta_min = self.config.get('eta_min', 0)
        T_max = self.config.get('cosine_t_max', 50)
        max_lr = self.config.get('onecycle_max_lr', 1e-3)
        div_factor = self.config.get('onecycle_div_factor', 25.0)
        final_div_factor = self.config.get('onecycle_final_div_factor', 10000.0)

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
