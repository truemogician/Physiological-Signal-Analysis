from typing import cast, Union, Tuple, Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader


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

def train_model(
    model: nn.Module, 
    train_iter: DataLoader[Tuple[Tensor, Tensor]], 
    optimizer: torch.optim.Optimizer, 
    criterion: Callable[[Tensor, Tensor], Tensor],
    iteration_num = 200,
    test_iter: Optional[DataLoader[Tuple[Tensor, Tensor]]] = None):
    result = dict(
        train_loss=[],
        train_acc=[]
    )
    if test_iter is not None:
        result["test_loss"] = []
        result["test_acc"] = []
    device = get_device()
    model = model.to(device)

    for iter in range(iteration_num):
        train_loss, train_acc, data_num = 0.0, 0, 0
        model.train()
        for data, label in train_iter:
            # 预测
            # emg = torch.squeeze(emg, dim=1)
            # eeg = torch.cat((eeg, emg), 1)
            data = cast(Tensor, data.to(torch.float32).to(device))
            label = cast(Tensor, label.to(device))
            out = cast(Tensor, model(data))
            pred = torch.argmax(out, dim=-1)

            # 更新
            optimizer.zero_grad()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            # 记录参数
            batch_size = data.size(0)
            train_loss += loss * batch_size
            train_acc += torch.sum(pred == label.data)
            data_num += batch_size
        
        train_loss = train_loss/ data_num
        train_acc = train_acc / data_num
        result["train_loss"].append(train_loss.item())
        result["train_acc"].append(train_acc.item())
        
        if test_iter is None:
            print(f"[{iter:02d}] train_acc: {train_acc:.4f}, train_loss: {train_loss:.2f}")
        else:
            test_loss, test_acc = run_model(model, test_iter, criterion)
            result["test_loss"].append(test_loss.item())
            result["test_acc"].append(test_acc.item())
            print("[{:02d}] train_acc: {:.4f}, test_acc: {:.4f}, train_loss: {:.2f}, test_loss: {:.2f}".format(
                iter,
                train_acc.item(),
                test_acc.item(),
                train_loss.item(),
                test_loss.item()
            ))
        
    return result