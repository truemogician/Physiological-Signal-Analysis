from typing import Tuple, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_device


def run_model(
    model: nn.Module, 
    test_iter: DataLoader[Tuple[Tensor, Tensor]], 
    criteria: Callable[[Tensor, Tensor], Tensor]):
    test_loss, test_acc, data_num = 0.0, 0, 0
    model.eval()
    device = get_device()
    model = model.to(device)

    for data, label in test_iter:
        data = data.to(device).to(torch.float32)
        label = label.to(device)
        out = model(data)
        pred = torch.argmax(out, dim=-1)
        loss = criteria(out, label)
        batch_size = data.size(0)
        test_loss += loss * batch_size
        data_num += batch_size
        test_acc += torch.sum(pred == label)
        
    test_loss = test_loss / data_num
    test_acc = test_acc / data_num
    return test_loss, test_acc