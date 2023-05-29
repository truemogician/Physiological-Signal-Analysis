import json
import sys
from typing import Tuple, Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.common import project_root, get_data_files
from utils.torch import get_device
from dataset.way_eeg_gal import WayEegGalDataset
from dataset.utils import create_data_loader


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
    data_files = get_data_files()
    model_file = args[0]
    indices = [int(i) for i in args[1:]]
    data_files = {k: v for k, v in data_files.items() if k in indices}
    
    model = torch.load(model_file)
    
    config: Dict = json.load(open(project_root / "config/motion_intention_detection.json", "r"))
    train_conf = config["train"]
    
    for subj, data_file in data_files.items():
        dataset = WayEegGalDataset(data_file)
        data, labels = dataset.prepare_for_motion_intention_detection()
        loader = create_data_loader(data, labels, batch_size=train_conf["batch_size"]) 
        loss, acc = run_model(
            model, 
            loader, 
            lambda out, target: nn.CrossEntropyLoss()(out, target.long())
        )
        print(f"[Subject {subj:02d}] Loss: {loss:.4f}, Acc: {acc:.4f}")