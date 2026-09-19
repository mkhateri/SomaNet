from typing import Callable, Tuple, Dict, Union
import shutil
import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def copy_logs(old_log_dir, new_log_dir):
    if os.path.exists(old_log_dir):
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
        for filename in os.listdir(old_log_dir):
            src_path = os.path.join(old_log_dir, filename)
            dst_path = os.path.join(new_log_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
        print(f"Copied logs from {old_log_dir} to {new_log_dir}")
    else:
        print(f"Old log directory {old_log_dir} does not exist.")


def makedirs_fn(*argv):
    path_ = [arg for arg in argv if arg]
    path_ = os.path.join(*path_)
    if not os.path.exists(path_):
        os.makedirs(path_)
        #tf.gfile.MakeDirs(path_)
    return path_