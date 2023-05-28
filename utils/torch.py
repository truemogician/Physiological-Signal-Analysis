from typing import Union

import torch


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