"""
This script reads and processes image data for inference in neural networks.
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

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Any, Dict, Tuple, Union
from utils.transform_collections import apply_inference_augmentation
import random

def _get_dir(_dir: Union[list, str]) -> list:
    """Clean and convert input directory path(s) into a list of directories."""
    if isinstance(_dir, str):
        _dir = [_dir]

    dirs = []
    for item in _dir:
        dirs.append("".join(list(map(lambda c: c if c not in r'[,*?!:"<>|] \\' else '', item))))
    return dirs

def _initilize_data_mode(data_mode: Dict[str, bool]) -> Dict[str, Any]:
    """Initialize data mode with default values and update with input data mode."""
    data_mode_ = {key: False for key in ['img', 'mask', 'prompt']}

    if data_mode is not None:
        data_mode_.update(data_mode)
    return data_mode_

def _get_files(data_dirs, img_format, data_mode):
    """Retrieve file paths for images based on the provided directories and data mode."""

    def _check_file_consistency(files: Dict[str, list]):
        """Check if the number of files in input and target directories correspond to each other."""
        if 'img' not in files:
            raise ValueError("There is no img file directory.")

        if 'mask' in files and len(files['img']) != len(files['mask']):
            raise ValueError("Mismatch between the number of input and mask files.")

        return True

    def _get_unique_path(input_list):
        """Return unique values from the input list."""
        return list(set(input_list))

    data = _initilize_data_mode(data_mode)
    data_dirs = [os.path.join(dir_, 'img') for dir_ in data_dirs]

    data['img'] = [os.path.join(path, name)
                   for data_dir in data_dirs
                   for path, subdirs, files_ in os.walk(data_dir)
                   for name in files_
                   if name.lower().endswith(tuple(img_format))]

    data['img'] = sorted(_get_unique_path(data['img']))

    if data_mode['mask']:
        data['mask'] = sorted([files.replace('/img/', '/mask/') for files in data['img']])

    if _check_file_consistency(data):
        return data
    

def _imRead(PATH):
    """Read an image from the given path and convert it to a NumPy array with an added channel dimension."""
    image = Image.open(PATH).convert('L')
    image_array = np.array(image)
    return np.expand_dims(image_array[:, :], axis=0)  # Adjust the cropping as needed

class Dataset_cls(Dataset):
    def __init__(self,
                 data_dirs: str,
                 data_mode: Dict[str, bool],
                 task_mode: str,
                 img_format: list,
                 crop_size: int):
        super(Dataset_cls, self).__init__()

        self.data_mode = _initilize_data_mode(data_mode)
        self.task_mode = task_mode
        self.files_ = _get_files(data_dirs, img_format, self.data_mode)
        self.crop = crop_size
        self.transform = apply_inference_augmentation()

    def __len__(self):
        return len(self.files_['img'])

    def __getitem__(self, index):
        """Retrieve a single data sample based on the index."""
        sample_index = _initilize_data_mode(self.data_mode)

        if self.data_mode['img']:
            sample_index['img'] = _imRead(self.files_['img'][index])  # Put normalization in the transforms 

        if self.data_mode['mask']:
            sample_index['mask'] = _imRead(self.files_['mask'][index])

        sample_index.update({'img_base_name': os.path.basename(self.files_['img'][index])})

        if self.task_mode == 'SEG':
            
            sample_index = self.transform(sample_index)  # Please check augmentation yourself with original implementations

        return sample_index

def custom_collate_fn(batch):
    """Custom collate function to handle empty samples in the batch."""
    batch = [item for item in batch if item is not None]  # Filter out None items
    if len(batch) == 0:
        return None  # Return None to indicate empty batch
    return torch.utils.data.dataloader.default_collate(batch)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

class DataLoader_cls:
    def __init__(self,
                 test_dir: Union[str, list],
                 data_mode: bool,
                 task_mode: 'str',
                 img_format: Union[str, list],
                 num_workers: int,
                 drop_last: bool,
                 pin_memory: bool):
        
        self.test_dir = _get_dir(test_dir)

        self.DL_params = {'num_workers': num_workers,
                          'pin_memory': pin_memory,
                          'drop_last': drop_last,
                          'worker_init_fn': worker_init_fn}  # Add worker_init_fn here

        self.DS_params = {'data_mode': data_mode,
                          'task_mode': task_mode,
                          'img_format': img_format}

    def _get_DS(self): 
        """Create and return data loaders for inference."""
        Dataset_Test = Dataset_cls(data_dirs=self.test_dir, data_mode=self.DS_params['data_mode'],
                                   task_mode=self.DS_params['task_mode'], img_format=self.DS_params['img_format'], 
                                   crop_size=self.DS_params.get('crop_size', 1024))
        
        inference_loader = DataLoader(dataset=Dataset_Test, shuffle=False, batch_size=1, collate_fn=custom_collate_fn, **self.DL_params)

        return {'inference_loader': inference_loader}
    
    @staticmethod
    def print_info(DL):
        print("DataLoader Information:")
        for name, loader in DL.items():
            print(f"Number of batches in {name}: {len(loader)}")

