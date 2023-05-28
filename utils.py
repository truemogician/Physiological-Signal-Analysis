import os
import re
from typing import Union

import torch

def get_data_files(data_dir: str = "data"):
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files: dict[int, str] = dict()
    for f in all_files:
        match = re.match(r"^ws_subj(\d+)\.npy$", f)
        if match:
            files[int(match.group(1))] = os.path.join(data_dir, f)
    return files


device: Union[torch.device, None] = None
def get_device():
    if device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
            device = gpus[0]
            free_memory = torch.cuda.mem_get_info(device)[0]
            for gpu in gpus[1:]:
                free_mem = torch.cuda.mem_get_info(gpu)[0]
                if free_mem > free_memory:
                    device = gpu
                    free_memory = free_mem
    return device