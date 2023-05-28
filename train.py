import time
import sys
from typing import Tuple

from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import xlrd
import xlwt

from model.GcnNet import GcnNet
from preprocess_data import *
from connectivity.PMI import *
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
    print(f"test_loss:{running_loss.item():.5f},test_acc:{running_acc.item():.5f}")

    return running_loss, running_acc


def train(
    subj: int, 
    model: GcnNet, 
    train_iter: DataLoader[Tuple[Tensor, Tensor]], 
    test_iter: DataLoader[Tuple[Tensor, Tensor]], 
    optimizer: torch.optim.Optimizer, 
    criteria, 
    epochs_num=200):
    result = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    device = get_device()
    model = model.to(device)
    print(next(model.parameters()).device)
    max_acc = 0.0

    for epoch in range(epochs_num):
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
        print("{} epoch,train_loss:{:.5f},train_acc:{:.5f}".format(
            epoch,
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
    sheet = workbook.add_sheet(f"sheet_{epoch}")
    data = list(result["train_loss"])
    for row in range(len(data)):
        sheet.write(row, 0, data[row])
    data = list(result["train_acc"])
    for row in range(len(data)):
        sheet.write(row, 1, data[row])
    data = list(result["test_loss"])
    for row in range(len(data)):
        sheet.write(row, 2, data[row])
    data = list(result["test_acc"])
    for row in range(len(data)):
        sheet.write(row, 3, data[row])
    workbook.save(f"result/sub-{subj:02d}/train_stats.xls")
    
    # 画出acc和loss的曲线
    plt.figure()
    plt.plot(result["train_loss"], label="train_loss")
    plt.plot(result["test_loss"], label="test_loss")
    plt.legend()
    plt.savefig(f"result/sub-{subj:02d}/loss_l2_0.001_filter_005_50.png")
    plt.figure()
    plt.plot(result["train_acc"], label="train_acc")
    plt.plot(result["test_acc"], label=f"test_acc,mean={test_acc_mean:.5f}")
    plt.legend()
    plt.savefig(f"result/sub-{subj:02d}/acc_l2_0.001_filter_005_50.png")
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
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    for [subj, data_file] in data_files.items():
        print(f"Subject {subj}")
        start_time = time.time()
        adj_path = f"result/sub-{subj:02d}/eeg_initial_weight.xls"
        data = xlrd.open_workbook(adj_path)
        # adj_mat_array = np.empty((0, 32, 32))
        adj_mat_array = []
        for i in range(4):
            temp = []
            table = data.sheets()[i]
            for j in range(table.nrows):
                temp.append(table.row_values(j))
            adj_mat_array.append(temp)
        # 将上三角矩阵转换为对称矩阵
        for i in range(len(adj_mat_array)):
            for j in range(len(adj_mat_array[i])):
                for k in range(0, j):
                    adj_mat_array[i][j][k] = adj_mat_array[i][k][j]

        # 随机初始化邻接矩阵为0~2之间的数
        # for i in range(len(adj_mat_array)):
        #     for j in range(len(adj_mat_array[i])):
        #         for k in range(len(adj_mat_array[i][j])):
        #             adj_mat_array[i][j][k] = random.uniform(0, 2)

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("sheet1")
        for row in range(len(adj_mat_array[0])):
            for col in range(len(adj_mat_array[0][0])):
                sheet.write(row, col, round(adj_mat_array[0][row][col], 5))

        workbook.save(f"result/sub-{subj:02d}/pre_connectivity_matrix.xls")
        adj_mat_array = np.array(adj_mat_array, dtype=np.float32)

        # GCN_NET model
        gcn_net_model = GcnNet(
            node_emb_dims=800,
            adj_mat_array=adj_mat_array[0],
            num_classes=3,
        )
        train_iter, test_iter = get_data_check_intend(data_file)
        optimizer = torch.optim.Adam(gcn_net_model.parameters(), lr=0.0001, weight_decay=1e-3)

        train(
            subj=subj,
            model=gcn_net_model,
            train_iter=train_iter,
            test_iter=test_iter,
            optimizer=optimizer,
            # 交叉熵损失函数加上l2正则化
            criteria=focal_loss_with_regularization,  # nn.CrossEntropyLoss(),
            epochs_num=200,
        )
        print(f"Time Cost:{time.time() - start_time:.5f}s")
        trained_adj = gcn_net_model.get_matrix()
        # 将trained_adj保存到excel中
        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet("sheet1")
        for row in range(trained_adj.shape[0]):
            for col in range(trained_adj.shape[1]):
                sheet.write(row, col, trained_adj[row][col].item())

        workbook.save(f"result/sub-{subj:02d}/trained_connectivity_matrix.xls")
