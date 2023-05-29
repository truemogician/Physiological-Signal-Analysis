import json
import sys
from typing import Tuple, Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.torch import get_device
from preprocess_data import get_data_motion_intention


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

if __name__ == "__main__":
    args = sys.argv[1:]
    model_file = args[0]
    data_file = args[1]
    
    model = torch.load(model_file)
    
    config: Dict = json.load(open("config/motion_intention.json", "r"))
    train_conf = config["train"]
    data_iter, _ = get_data_motion_intention(
        data_file,
        batch_size=train_conf["batch_size"],
        test_size=0
    )
    
    loss, acc = run_model(
        model, 
        data_iter, 
        lambda out, target: nn.CrossEntropyLoss()(out, target.long())
    )
    print(f"Loss: {loss:.4f}, Acc: {acc:.4f}")