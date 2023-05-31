import time
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import xlwt

from model.GcnNet import GcnNet
from model.utils import train_model, run_model
from dataset.way_eeg_gal import Dataset
from dataset.utils import create_data_loader, create_train_test_loader
from utils.common import project_root, get_data_files, load_config, ensure_dir
from utils.visualize import NodeMeta, PlotStyle, visualize_matrix
from connectivity.PMI import SPMI_1epoch


exp_name = Path(__file__).stem
config = load_config(exp_name)
path_conf = config["path"]

def train(
    data_file: Union[os.PathLike, List[os.PathLike]],
    result_dir: os.PathLike, 
    allow_cache = True, 
    batch = 1):   
    data_files = data_file if isinstance(data_file, list) else [data_file]
    dataset = Dataset(data_files[0], allow_cache=allow_cache)
    eeg, labels = dataset.prepare_for_motion_intention_detection(config["data"]["interval"])
    matrix_trial = eeg[0]
    for df in data_files[1:]:
        dataset = Dataset(df, allow_cache=allow_cache)
        eeg_, labels_ = dataset.prepare_for_motion_intention_detection(config["data"]["interval"])
        eeg = np.concatenate((eeg, eeg_), axis=0)
        labels = np.concatenate((labels, labels_), axis=0)
        matrix_trial = np.concatenate((matrix_trial, eeg_[0]), axis=1)
    
    # 初始化关联性矩阵
    initial_matrix_path = ensure_dir(result_dir) / path_conf["initial_matrix"]
    if os.path.exists(initial_matrix_path):
        matrix = np.loadtxt(initial_matrix_path, delimiter=",")
    else:
        matrix = SPMI_1epoch(matrix_trial, 6, 2)
        matrix_mask = np.full(matrix.shape, True, dtype=bool)
        np.fill_diagonal(matrix_mask, False)
        min = matrix.min(initial=sys.float_info.max, where=matrix_mask)
        max = matrix.max(initial=sys.float_info.min, where=matrix_mask)
        rescaled_min, rescaled_max = min / 2, max / 2 + 0.5
        matrix = (matrix - min) / (max - min) * (rescaled_max - rescaled_min) + rescaled_min
        np.savetxt(ensure_dir(initial_matrix_path), matrix, fmt="%.6f", delimiter=",")

    # 随机初始化邻接矩阵为0~1之间的数
    # matrix = np.random.rand(32, 32).astype(np.float32)
    # for i in range(matrix.shape[0]):
    #    for j in range(0, i):
    #        matrix[i, j] = matrix[j, i]

    model_conf = config["model"]
    train_conf = config["train"]
    min_loss = sys.float_info.max
    max_acc = 0
    stats_workbook = xlwt.Workbook()
    training_results = []
    for idx in range(batch):
        if batch > 1:
            print(f"Training the {idx + 1}{['st', 'nd', 'rd'][idx] if idx < 3 else 'th'} model...")
        # 构建模型
        gcn_net_model = GcnNet(
            node_embedding_dims=model_conf["node_embedding_dim"],
            class_num=model_conf["class_num"],
            adjacent_matrix=matrix
        )
        # 训练模型
        iter_num = train_conf["iteration_num"]
        train_iter, test_iter = create_train_test_loader(
            eeg,
            labels,
            batch_size=train_conf["batch_size"],
            test_size=train_conf["test_size"]
        )
        optimizer = torch.optim.Adam(
            gcn_net_model.parameters(),
            lr=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"]
        )
        start_time = time.time()
        result = train_model(
            gcn_net_model, 
            train_iter, 
            optimizer, 
            lambda out, target: nn.CrossEntropyLoss()(out, target.long()), 
            iter_num, 
            test_iter
        )
        print(f"Training time: {time.time() - start_time:.2f}s")
        training_results.append(result)
        
        sheet = stats_workbook.add_sheet(f"model-{idx}")
        for col, header in enumerate(["train_loss", "train_acc", "test_loss", "test_acc"]):
            sheet.write(0, col, header)
            for row, value in enumerate(result[header]):
                sheet.write(row + 1, col, value)
        
        if result["test_loss"][-1] < min_loss:
            min_loss = result["test_loss"][-1]
            best_model = gcn_net_model
            best_model_idx = idx
            best_result = result
            
        if result["test_acc"][-1] > max_acc:
            max_acc = result["test_acc"][-1]
            max_acc_model = gcn_net_model
            max_acc_model_idx = idx
            max_acc_result = result
    
    # 保存训练统计数据
    stats_workbook.save(ensure_dir(result_dir / path_conf["training_stats"]))
    
    if batch > 1:
        stats_workbook.get_sheet(best_model_idx).name += " (min_loss)"
        stats_workbook.get_sheet(max_acc_model_idx).name += " (max_acc)"
        print(f"Min Loss: test_acc={best_result['test_acc'][-1]:.4f}, test_loss={best_result['test_loss'][-1]:.4f}")
        print(f"Max Acc: test_acc={max_acc_result['test_acc'][-1]:.4f}, test_loss={max_acc_result['test_loss'][-1]:.4f}")
        avg_acc = np.mean([result["test_acc"][-1] for result in training_results])
        avg_loss = np.mean([result["test_loss"][-1] for result in training_results])
        print(f"Average: test_acc={avg_acc:.4f}, test_loss={avg_loss:.4f}") 
    
    # 画出最佳模型的acc和loss的曲线
    plt.figure()
    plt.plot(best_result["train_loss"], label="train_loss")
    plt.plot(best_result["test_loss"], label="test_loss")
    plt.legend()
    plt.savefig(ensure_dir(result_dir / path_conf["loss_plot"]))
    plt.figure()
    plt.plot(best_result["train_acc"], label="train_acc")
    plt.plot(best_result["test_acc"], label=f"test_acc")
    plt.legend()
    plt.savefig(ensure_dir(result_dir / path_conf["acc_plot"]))
    
    # 保存最佳模型的关联性矩阵
    trained_matrix = best_model.get_matrix().cpu() 
    np.savetxt(ensure_dir(result_dir / path_conf["trained_matrix"]), trained_matrix, fmt="%.6f", delimiter=",")
    
    # 最佳模型关联性矩阵可视化
    matrix_plot_styles = config["plot"]["matrix"]["styles"]
    metadata = [
        NodeMeta(n, v.coordinate, v.group)
        for n, v in Dataset.eeg_electrode_metadata.items()
    ]
    figure = visualize_matrix(
        trained_matrix.numpy(), 
        metadata, 
        style=PlotStyle(
            node=matrix_plot_styles["node"],
            font=matrix_plot_styles["font"],
            layout=matrix_plot_styles["layout"]
        )
    )
    figure.write_html(ensure_dir(result_dir / path_conf["matrix_plot"]))
    
    # 保存最佳模型
    torch.save(best_model, ensure_dir(result_dir / path_conf["model"]))
 
def run(model_file: os.PathLike, data_files: List[os.PathLike]):
    model = torch.load(model_file)
    for data_file in data_files:
        dataset = Dataset(data_file)
        data, labels = dataset.prepare_for_motion_intention_detection(config["data"]["interval"])
        loader = create_data_loader(data, labels, batch_size=config["train"]["batch_size"]) 
        loss, acc = run_model(
            model, 
            loader, 
            lambda out, target: nn.CrossEntropyLoss()(out, target.long())
        )
        print(f"[{Path(data_file).name}] Loss: {loss:.4f}, Acc: {acc:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train or run model")
    sub_parasers = parser.add_subparsers(dest="command")
    train_parser = sub_parasers.add_parser("train", help="Train model")
    train_parser.add_argument("subject_indices", nargs="+", help="Indices of subjects whose data will be used for training")
    train_parser.add_argument("--no-cache", action="store_true", help="Whether to use cache")
    train_parser.add_argument("--batch", type=int, default="1", help="Time the model will be trained")
    train_parser.add_argument("--result_dir", help="Path to directory where results will be saved")
    run_parser = sub_parasers.add_parser("run", help="Run model")
    run_parser.add_argument("model_file", help="Path to model file")
    run_parser.add_argument("subject_indices", nargs="+", help="Indices of subjects whose data will be used for running")
    args = parser.parse_args()

    data_files = get_data_files()
    if args.command == "train":
        indices = [int(i) for i in args.subject_indices]
        indices.sort()
        valid_indices = data_files.keys()
        if any([i not in valid_indices for i in indices]):
            raise ValueError("Invalid subject index")
        data_files = {i: data_files[i] for i in indices}
        result_dir = args.result_dir if args.result_dir else project_root / f"result/sub-{'+'.join([str(i).zfill(2) for i in indices])}" / exp_name
        train(list(data_files.values()), result_dir, not args.no_cache, args.batch)
    if args.command == "run":
        indices = [int(i) for i in args.subject_indices]
        data_files = [v for k, v in data_files.items() if k in indices]
        run(args.model_file, data_files)