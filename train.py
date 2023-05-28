import time
import sys
import os
import json
from typing import Tuple, Dict

from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import xlrd
import xlwt

from model.GcnNet import GcnNet
from preprocess_data import *
from connectivity.PMI import *
from compute_weight import compute_weight
from visualize import NodeMeta, PlotStyle, visualize
from utils.common import get_data_files
from utils.torch import get_device


def test(model, test_iter, criteria):
    running_loss, running_corrects, data_num = 0.0, 0, 0
    model.eval()
    device = get_device()
    model = model.to(device)

    for eeg, label in test_iter:
        # emg = torch.squeeze(emg, dim=1)
        # eeg = torch.cat((eeg, emg), 1)
        eeg = eeg.to(device).to(torch.float32)
        label = label.to(device)
        out = model(eeg)

        pred = torch.argmax(out, dim=-1)
        loss = criteria(model, out, label)
        batch_size = eeg.size(0)
        running_loss += loss * batch_size
        data_num += batch_size
        running_corrects += torch.sum(pred == label)
    running_loss = running_loss / data_num
    running_acc = running_corrects / data_num
    print(f"[Test]  loss: {running_loss.item():.4f}, acc: {running_acc.item():.4f}")

    return running_loss, running_acc


def train(
    subj: int, 
    model: GcnNet, 
    train_iter: DataLoader[Tuple[Tensor, Tensor]], 
    test_iter: DataLoader[Tuple[Tensor, Tensor]], 
    optimizer: torch.optim.Optimizer, 
    criteria, 
    iteration_num=200):
    result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    device = get_device()
    print(f"Using device {device}")
    model = model.to(device)
    max_acc = 0.0

    for iter in range(iteration_num):
        running_loss, running_corrects, data_num = 0.0, 0, 0
        model.train()
        for eeg, label in train_iter:
            # 预测
            # emg = torch.squeeze(emg, dim=1)
            # eeg = torch.cat((eeg, emg), 1)
            eeg = eeg.to(device).to(torch.float32)
            label = label.to(device)
            out = model(eeg)
            pred = torch.argmax(out, dim=-1)

            # 更新
            optimizer.zero_grad()
            loss = criteria(model, out, label)
            loss.backward()
            optimizer.step()

            # 记录参数
            batch_size = eeg.size(0)
            running_loss += loss * batch_size
            running_corrects += torch.sum(pred == label.data)
            data_num += batch_size
        # if (epoch+1) % 10 == 0:
        #     graph_weight = model.get_weight()
        # print(graph_weight)
        print("{} iteration\n[Train] loss: {:.4f}, acc: {:.4f}".format(
            iter,
            (running_loss / data_num).item(),
            (running_corrects / data_num).item()
        ))
        result["train_loss"].append((running_loss / data_num).item())
        result["train_acc"].append((running_corrects / data_num).item())
        test_loss, test_acc = test(model, test_iter, criteria, device)
        result["test_loss"].append(test_loss.item())
        result["test_acc"].append(test_acc.item())
        # 把result保存到excel中
        if max_acc < test_acc:
            max_acc = test_acc
    
    length = len(result["test_acc"])
    test_acc_mean = sum(result["test_acc"][length // 2 :]) / len(result["test_acc"][length // 2 :])
    result["test_acc"].append(test_acc_mean)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(f"sheet_{iter}")
    sheet.write(0, 0, "train_loss")
    sheet.write(0, 1, "train_acc")
    sheet.write(0, 2, "test_loss")
    sheet.write(0, 3, "test_acc")
    for i in range(len(result["train_loss"])):
        sheet.write(i + 1, 0, result["train_loss"][i])
        sheet.write(i + 1, 1, result["train_acc"][i])
        sheet.write(i + 1, 2, result["test_loss"][i])
        sheet.write(i + 1, 3, result["test_acc"][i])
    workbook.save(f"result/sub-{subj:02d}/train_stats.xlsx")
    
    # 画出acc和loss的曲线
    plt.figure()
    plt.plot(result["train_loss"], label="train_loss")
    plt.plot(result["test_loss"], label="test_loss")
    plt.legend()
    plt.savefig(f"result/sub-{subj:02d}/{config['plot']['loss']['filename']}")
    plt.figure()
    plt.plot(result["train_acc"], label="train_acc")
    plt.plot(result["test_acc"], label=f"test_acc,mean={test_acc_mean:.5f}")
    plt.legend()
    plt.savefig(f"result/sub-{subj:02d}/{config['plot']['accuracy']['filename']}")
    return max_acc


# L2 正则化
def L2Loss(model: GcnNet, alpha: float):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if "bias" not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss


def L1Loss(model: GcnNet, beta: float):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if "bias" not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss


def focal_loss_with_regularization(model: GcnNet, y_pred, y_true):
    focal = nn.CrossEntropyLoss()(y_pred, y_true.long())
    # l1_loss = L1Loss(model, 0.001)
    # l2_loss = L2Loss(model, 0.001)
    total_loss = focal  # + l2_loss  # + l1_loss
    return total_loss


if __name__ == "__main__":
    data_files = get_data_files()
    config: Dict = json.load(open("config/motion_intention.json", "r"))
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    for [subj, data_file] in data_files.items():
        print(f"Training model using data from subject {subj}...")
        start_time = time.time()
        adj_path = f"result/sub-{subj:02d}/eeg_initial_weight.xlsx"
        if not os.path.exists(adj_path):
            compute_weight(data_file, adj_path)
        workbook = xlrd.open_workbook(adj_path)
        
        adjacent_matrixs = []
        for sheet in workbook.sheets():
            matrix = [sheet.row_values(i) for i in range(sheet.nrows)]
            adjacent_matrixs.append(matrix)
        
        # 将上三角矩阵转换为对称矩阵
        for k in range(len(adjacent_matrixs)):
            for i in range(len(adjacent_matrixs[k])):
                for j in range(0, i):
                    adjacent_matrixs[k][i][j] = adjacent_matrixs[k][j][i]

        # 随机初始化邻接矩阵为0~2之间的数
        # for i in range(len(adj_mat_array)):
        #     for j in range(len(adj_mat_array[i])):
        #         for k in range(len(adj_mat_array[i][j])):
        #             adj_mat_array[i][j][k] = random.uniform(0, 2)

        workbook = xlwt.Workbook()
        for k in range(len(adjacent_matrixs)):
            sheet = workbook.add_sheet(f"trial_{k}")
            for row in range(len(adjacent_matrixs[k])):
                for col in range(len(adjacent_matrixs[k][row])):
                    sheet.write(row, col, round(adjacent_matrixs[k][row][col], 5))

        workbook.save(f"result/sub-{subj:02d}/pre_connectivity_matrix.xlsx")
        adjacent_matrixs = np.array(adjacent_matrixs, dtype=np.float32)

        # GCN_NET model
        model_conf = config["model"]
        gcn_net_model = GcnNet(
            node_embedding_dims=model_conf["node_embedding_dim"],
            adj_mat_array=adjacent_matrixs[0],
            class_num=model_conf["class_num"],
        )
        
        train_conf = config["train"]
        train_iter, test_iter = get_data_motion_intention(
            data_file,
            batch_size=train_conf["batch_size"],
            test_size=train_conf["test_size"]
        )
        optimizer = torch.optim.Adam(
            gcn_net_model.parameters(),
            lr=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"]
        )
        train(
            subj=subj,
            model=gcn_net_model,
            train_iter=train_iter,
            test_iter=test_iter,
            optimizer=optimizer,
            criteria=focal_loss_with_regularization,  # nn.CrossEntropyLoss(),
            iteration_num=train_conf["iteration_num"]
        )
        print(f"Time Cost :{time.time() - start_time:.5f}s")
        
        conn_plot_conf = config["plot"]["connectivity"]
        trained_adj = gcn_net_model.get_matrix()
        # 将trained_adj保存到excel中
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("trial_0")
        for row in range(trained_adj.shape[0]):
            for col in range(trained_adj.shape[1]):
                sheet.write(row, col, trained_adj[row][col].item())
        workbook.save(f"result/sub-{subj:02d}/trained_connectivity_matrix.xlsx")
        metadata = [NodeMeta(n, v.coordinate, v.group) for n, v in eeg_electrode_metadata.items()]
        figure = visualize(
            trained_adj.cpu().numpy(), 
            metadata, 
            style=PlotStyle(
                node=conn_plot_conf["styles"]["node"],
                font=conn_plot_conf["styles"]["font"],
                layout=conn_plot_conf["styles"]["layout"]
            )
        )
        figure.write_html(f"result/sub-{subj:02d}/{conn_plot_conf['filename']}")