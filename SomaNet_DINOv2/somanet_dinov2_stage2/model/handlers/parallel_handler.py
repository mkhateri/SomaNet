import torch.nn as nn

class ParallelHandler:
    @staticmethod
    def apply_parallel(model: nn.Module, num_gpus: int) -> nn.Module:
        if num_gpus > 1:
            return nn.DataParallel(model)
        return model

    @staticmethod
    def adjust_state_dict(state_dict: dict, num_gpus: int) -> dict:
        if len(state_dict) == 0:
            raise ValueError("State dictionary is empty. Please provide a valid state dictionary.")

        # Handle case when num_gpus > 1 but the state_dict keys don't have 'module.' prefix
        if num_gpus > 1 and not list(state_dict.keys())[0].startswith('module.'):
            return {f"module.{k}": v for k, v in state_dict.items()}

        # Handle case when num_gpus == 1 but the state_dict keys have 'module.' prefix
        elif num_gpus == 1 and list(state_dict.keys())[0].startswith('module.'):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # If state_dict is already correctly formatted, return it as is
        return state_dict
