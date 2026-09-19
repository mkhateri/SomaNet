"""
This script reads and processes image data for training and testing neural networks.
"""
#####################################################################################
# data_dir1/                      data_dir2/          ...            data_dirN/
# ├── img/                        ├── img/                           ├── img/
# │   ├── img1.png                │   ├── img1.png                   │   ├── img1.png
# │   ├── img2.png                │   ├── img2.png                   │   ├── img2.png
# │   ├── img3.png                │   ├── img3.png                   │   ├── img3.png
# │   └── ...                     │   └── ...                        │   └── ...
# ├── mask/                       ├── mask/                          ├── mask/
# │   ├── img1.png                │   ├── img1.png                   │   ├── img1.png
# │   ├── img2.png                │   ├── img2.png                   │   ├── img2.png
# │   ├── img3.png                │   └── img3.png                   │   └── img3.png
# │   └── ...                     └── ...                            └── ...
# └── prompt/                     └── prompt/                        └── prompt/
#     ├── img1.png                    ├── img1.png                       ├── img1.png
#     ├── img2.png                    ├── img2.png                       ├── img2.png
#     ├── img3.png                    ├── img3.png                       ├── img3.png
#     └── ...                         └── ...                            └── ...
#####################################################################################
# dataloader_train.py

import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from PIL import Image
from typing import Any, Dict, Union
import random
from utils.affinity import multi_offset, gen_affs_ours, weight_binary_ratio  # Import from affinity.py
#from utils.transform_collections import apply_train_augmentation, apply_inference_augmentation
from utils.transform_collections import apply_basic_train_augmentation, apply_train_augmentation
from utils.utils import set_seed

def _get_dir(_dir: Union[list, str]) -> list:
    if isinstance(_dir, str):
        _dir = [_dir]
    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs

def _initilize_data_mode(data_mode: Dict[str, bool]) -> Dict[str, Any]:
    data_mode_ = {key: False for key in ['img', 'mask', 'prompt']}
    if data_mode is not None:
        data_mode_.update(data_mode)
    return data_mode_

def _get_files(data_dirs, img_format, data_mode):
    def _check_file_consistency(files: Dict[str, list]):
        if 'img' not in files:
            raise ValueError("There is no img file directory.")
        if 'mask' in files and len(files['img']) != len(files['mask']):
            raise ValueError("Mismatch between the number of input and mask files.")
        return True

    def _get_unique_path(input_list):
        return list(set(input_list))

    data = _initilize_data_mode(data_mode)
    data_dirs = [os.path.join(dir_, 'img') for dir_ in data_dirs]
    data['img'] = [os.path.join(path, name)
                   for data_dir in data_dirs
                   for path, subdirs, files_ in os.walk(data_dir)
                   for name in files_
                   if name.lower().endswith(tuple(img_format))]
    
    data['img'] = _get_unique_path(data['img'])

    if data_mode['mask']:
        data['mask'] = [files.replace('/img/', '/mask/') for files in data['img']]

    if _check_file_consistency(data):
        return data

def _imRead(PATH):
    image = Image.open(PATH).convert('L')
    image_array = np.array(image)
    return np.expand_dims(image_array[:, :], axis=0)

class Dataset_cls(Dataset):
    def __init__(self,
                 data_dirs: str,
                 Training: bool,
                 data_mode: Dict[str, bool],
                 task_mode: str,
                 img_format: list,
                 crop_size: int,
                 padding_size: int,
                 mask_nonzero_ratio_threshold: float,
                 separate_weight_affinity: bool,
                 neighbor_affinity: int,
                 shifts_affinity: list
                 ):
                 
        super(Dataset_cls, self).__init__()

        self.data_mode = _initilize_data_mode(data_mode)
        self.task_mode = task_mode
        self.Training = Training
        self.files_ = _get_files(data_dirs, img_format, self.data_mode)
        self.mask_nonzero_ratio_threshold = mask_nonzero_ratio_threshold
        self.crop_size = crop_size
        #self.transform = apply_train_augmentation(crop_size, padding_size)
        self.separate_weight_affinity = separate_weight_affinity  # Store separate_weight flag
        self.augmentation_basic = apply_basic_train_augmentation(crop_size, padding_size)
        # Define offsets for affinity calculation
        self.offsets = multi_offset(shifts=shifts_affinity, neighbor=neighbor_affinity)

    def __len__(self):
        return len(self.files_['img'])

    def _is_valid_sample(self, sample_index):
        image_tensor = sample_index['img']
        mask_tensor = sample_index['mask']
        
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
        if isinstance(mask_tensor, np.ndarray):
            mask_tensor = torch.from_numpy(mask_tensor)
        
        return not torch.all(image_tensor == 0) and not torch.all(mask_tensor == 0)


    def constraint_checking(self, sample_index, mask_threshold, img_threshold=0.8):
        # Ensure 'mask' is a tensor (convert if it's a NumPy array)
        if isinstance(sample_index['mask'], np.ndarray):
            mask_tensor = torch.from_numpy(sample_index['mask'])
        else:
            mask_tensor = sample_index['mask']

        # Ensure 'img' is a tensor (convert if it's a NumPy array)
        if isinstance(sample_index['img'], np.ndarray):
            img_tensor = torch.from_numpy(sample_index['img'])
        else:
            img_tensor = sample_index['img']

        # Calculate the non-zero ratio for mask and image
        non_zero_ratio_mask = torch.sum(mask_tensor > 0).item() / mask_tensor.numel()
        non_zero_ratio_img = torch.sum(img_tensor > 0).item() / img_tensor.numel()

        # Return True if both conditions meet the specified thresholds
        return (non_zero_ratio_mask > mask_threshold) and (non_zero_ratio_img > img_threshold)
    
    def __getitem__(self, index):
        sample_index = _initilize_data_mode(self.data_mode)

        if self.data_mode['img']:
            sample_index['img'] = _imRead(self.files_['img'][index])

        if self.data_mode['mask']:
            sample_index['mask'] = _imRead(self.files_['mask'][index])

        if self.data_mode['prompt']:
            sample_index['prompt'] = _imRead(self.files_['prompt'][index])

        if self.task_mode == 'SEG':                                         
            sample_index = self.augmentation_basic(sample_index)

        if self._is_valid_sample(sample_index) and self.constraint_checking(sample_index, self.mask_nonzero_ratio_threshold):


            sample_index =  apply_train_augmentation(sample_index,
                                                               self.offsets,
                                                               self.separate_weight_affinity,
                                                               mode='all',
                                                               )
            
            #sample_index = {'aug_supervised': augmented_samples_supervised, 'aug_weakly_supervised': augmented_samples_weakly_supervised}


            return sample_index
        else:
            return self.__getitem__((index + 1) % len(self))

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None] 
    if len(batch) == 0:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"Error in collate function: {e}")
        print(f"Batch: {batch}")
        raise e

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

class DataLoader_cls:
    def __init__(self,
                 train_dir: Union[str, list],
                 data_mode: bool,
                 task_mode: 'str',
                 img_format: Union[str, list],
                 val_split: float,
                 batch_size: int,
                 crop_size: int,
                 padding_size: int,
                 mask_nonzero_ratio_threshold: float,
                 separate_weight_affinity: bool,
                 neighbor_affinity: int,
                 shifts_affinity: list,
                 num_workers: int,
                 drop_last: bool,
                 pin_memory: bool,
                 random_seed: int,
                 shuffle: bool):

        set_seed(random_seed)
        self.train_dir = _get_dir(train_dir)
        self.val_split = val_split
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.DL_params = {'num_workers': num_workers,
                          'pin_memory': pin_memory,
                          'drop_last': drop_last,
                          'worker_init_fn': worker_init_fn}

        self.DS_params = {'data_mode': data_mode,
                          'task_mode': task_mode,
                          'img_format': img_format,
                          'crop_size': crop_size,
                          'padding_size': padding_size,
                          'mask_nonzero_ratio_threshold': mask_nonzero_ratio_threshold,
                          'separate_weight_affinity': separate_weight_affinity,
                          'neighbor_affinity': neighbor_affinity,
                          'shifts_affinity': shifts_affinity}  # Pass separate_weight flag to dataset class



    def _get_DS(self):
        Dataset_Train = Dataset_cls(data_dirs=self.train_dir, Training=True, **self.DS_params)

        len_train_valid = Dataset_Train.__len__()
        train_sampler, valid_sampler = self._train_valid_sampler(len_train_valid)

        train_loader = DataLoader(dataset=Dataset_Train, sampler=train_sampler, batch_size=self.batch_size, collate_fn=custom_collate_fn, **self.DL_params)
        valid_loader = DataLoader(dataset=Dataset_Train, sampler=valid_sampler, batch_size=self.batch_size, collate_fn=custom_collate_fn, **self.DL_params)

        return {'train_loader': train_loader,
                'valid_loader': valid_loader}
    
    def _train_valid_sampler(self, len_train_valid):
        indices = list(range(len_train_valid))
        split = int(np.floor(self.val_split * len_train_valid))

        if self.shuffle:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler
    
    @staticmethod
    def print_info(DL):
        print("DataLoader Information:")
        for name, loader in DL.items():
            print(f"Number of batches in {name}: {len(loader)}")

