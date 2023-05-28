from typing import cast, Union

import torch


def get_device():
    device = cast(Union[torch.device, None], get_device.device)
    if device is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
            idx = 0
            free_memory = torch.cuda.mem_get_info(gpus[idx])[0]
            for i in range(1, len(gpus)):
                free_mem = torch.cuda.mem_get_info(gpus[i])[0]
                if free_mem > free_memory:
                    idx = i
                    free_memory = free_mem
            device = torch.device(f"cuda:{idx}")
        get_device.device = device
    return device
get_device.device = None