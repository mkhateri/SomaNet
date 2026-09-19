from typing import Tuple, Dict, Any, Union
from torch.utils.data import DataLoader
from utils.dataloader_train import DataLoader_cls as DataLoader_cls_train
from utils.dataloader_inference import DataLoader_cls as DataLoader_cls_inference 
from utils.utils import set_seed

class DataHandler:
    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.config = config
        set_seed(self.config['random_seed'])
    @staticmethod
    def print_info(DL: Dict[str, DataLoader]):
        """Print information about the DataLoader."""
        print("DataLoader Information:")
        for name, loader in DL.items():
            print(f"Number of batches in {name}: {len(loader)}")
        
    def _get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:            
        DS_params = {
            'train_dir': self.config['train_dir'],
            'data_mode': self.config['data_mode'],
            'task_mode': self.config['task_mode'],
            'img_format': self.config['img_format'],
            'val_split': self.config['val_split'],
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'drop_last': self.config['drop_last'],
            'pin_memory': self.config['pin_memory'],
            'random_seed': self.config['random_seed'],
            'shuffle': self.config['shuffle'],
            'crop_size': self.config['crop_size'], 
            'padding_size': self.config['padding_size'],
            'mask_nonzero_ratio_threshold': self.config['mask_nonzero_ratio_threshold'],
            'separate_weight_affinity': self.config['separate_weight_affinity'],
            'neighbor_affinity': self.config['neighbor_affinity'],           
            'shifts_affinity': self.config['shifts_affinity'],
        }

        DL = DataLoader_cls_train(**DS_params)._get_DS()

        self.print_info(DL)
        
        #return DL['train_loader'], DL['valid_loader'], DL['test_loader']
        return DL['train_loader'], DL['valid_loader']


    def _get_inference_dataloaders(self) -> Tuple[DataLoader]:
            
        DS_params = {
            'test_dir': self.config['test_dir'],
            'data_mode': self.config['data_mode'],
            'task_mode': self.config['task_mode'],
            'img_format': self.config['img_format'],
            'num_workers': self.config['num_workers'],
            'drop_last': self.config['drop_last'],
            'pin_memory': self.config['pin_memory'],
        }

        DL = DataLoader_cls_inference(**DS_params)._get_DS()

        self.print_info(DL)
        
        return DL['inference_loader']



