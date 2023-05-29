import time
import sys
import os
import json
from pathlib import Path
from typing import cast, Tuple, Dict, Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import xlrd
import xlwt

from model.GcnNet import GcnNet
from dataset.way_eeg_gal import WayEegGalDataset
from dataset.utils import create_train_test_loader
from utils.common import get_data_files, project_root
from utils.torch import get_device
from initialize_matrix import initialize_matrix
from run_model import run_model
from visualize import NodeMeta, PlotStyle, visualize


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


if __name__ == "__main__":
    config: Dict = json.load(open(project_root / "config/motion_intention_detection.json", "r"))
    data_files = get_data_files()
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    for [subj, data_file] in data_files.items():
        print(f"Training model using data from subject {subj}...")
        result_dir = project_root / f"result/sub-{subj:02d}"
        
        # 初始化关联性矩阵
        initial_matrix_path = result_dir / "initial_matrix.xlsx"
        if not os.path.exists(initial_matrix_path):
            initialize_matrix(data_file, initial_matrix_path)
        workbook = xlrd.open_workbook(initial_matrix_path)
        sheet = workbook.sheet_by_index(0)
        matrix = np.array([sheet.row_values(i) for i in range(sheet.nrows)], dtype=np.float32)

        # 随机初始化邻接矩阵为0~1之间的数
        # for i in range(len(adj_mat_array)):
        #     for j in range(len(adj_mat_array[i])):
        #         for k in range(len(adj_mat_array[i][j])):
        #             adj_mat_array[i][j][k] = random.uniform(0, 1)

        # GCN_NET model
        model_conf = config["model"]
        gcn_net_model = GcnNet(
            node_embedding_dims=model_conf["node_embedding_dim"],
            class_num=model_conf["class_num"],
            adjacent_matrix=matrix
        )
        
        train_conf = config["train"]
        dataset = WayEegGalDataset(data_file)
        train_iter, test_iter = create_train_test_loader(
            *dataset.prepare_for_motion_intention_detection(),
            batch_size=train_conf["batch_size"],
            test_size=train_conf["test_size"]
        )
        optimizer = torch.optim.Adam(
            gcn_net_model.parameters(),
            lr=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"]
        )

        # 训练模型
        iter_num = train_conf["iteration_num"]
        def loss_function(out: Tensor, target: Tensor):
            focal = nn.CrossEntropyLoss()(out, target.long())
            # l1_loss = L1Loss(model, 0.001)
            # l2_loss = L2Loss(model, 0.001)
            total_loss = focal  # + l2_loss  # + l1_loss
            return total_loss
        start_time = time.time()
        result = train_model(
            gcn_net_model, 
            train_iter, 
            optimizer, 
            loss_function, 
            iter_num, 
            test_iter
        )
        print(f"Training time: {time.time() - start_time:.2f}s")
        
        # 将训练结果保存到excel中
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet(f"sheet")
        sheet.write(0, 0, "train_loss")
        sheet.write(0, 1, "train_acc")
        sheet.write(0, 2, "test_loss")
        sheet.write(0, 3, "test_acc")
        for i in range(len(result["train_loss"])):
            sheet.write(i + 1, 0, result["train_loss"][i])
            sheet.write(i + 1, 1, result["train_acc"][i])
            sheet.write(i + 1, 2, result["test_loss"][i])
            sheet.write(i + 1, 3, result["test_acc"][i])
        workbook.save(result_dir / "train_stats.xlsx")
        
        # 画出acc和loss的曲线
        plt.figure()
        plt.plot(result["train_loss"], label="train_loss")
        plt.plot(result["test_loss"], label="test_loss")
        plt.legend()
        plt.savefig(result_dir / config["plot"]["loss"]["filename"])
        plt.figure()
        plt.plot(result["train_acc"], label="train_acc")
        plt.plot(result["test_acc"], label=f"test_acc")
        plt.legend()
        plt.savefig(result_dir / config["plot"]["accuracy"]["filename"])
        
        conn_plot_conf = config["plot"]["connectivity"]
        trained_matrix = gcn_net_model.get_matrix()
        
        # 将trained_matrix保存到excel中
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("trial_0")
        for row in range(trained_matrix.shape[0]):
            for col in range(trained_matrix.shape[1]):
                sheet.write(row, col, trained_matrix[row][col].item())
        workbook.save(result_dir / "trained_matrix.xlsx")
        
        # 画出关联性矩阵
        metadata = [
            NodeMeta(n, v.coordinate, v.group)
            for n, v in WayEegGalDataset.eeg_electrode_metadata.items()
        ]
        figure = visualize(
            trained_matrix.cpu().numpy(), 
            metadata, 
            style=PlotStyle(
                node=conn_plot_conf["styles"]["node"],
                font=conn_plot_conf["styles"]["font"],
                layout=conn_plot_conf["styles"]["layout"]
            )
        )
        figure.write_html(result_dir / conn_plot_conf["filename"])
        
        # 保存模型
        torch.save(gcn_net_model, result_dir / "model.pt")