import torch.nn as nn

class ParallelHandler:
    @staticmethod
    def apply_parallel(model: nn.Module, num_gpus: int) -> nn.Module:
        if num_gpus > 1:
            return nn.DataParallel(model)
        return model

    @staticmethod
    def adjust_state_dict(state_dict: dict, num_gpus: int) -> dict:
        if num_gpus > 1 and not list(state_dict.keys())[0].startswith('module.'):
            return {f"module.{k}": v for k, v in state_dict.items()}
        elif num_gpus == 1 and list(state_dict.keys())[0].startswith('module.'):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        return state_dict
