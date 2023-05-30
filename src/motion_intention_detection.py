import csv
import time
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

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

def train(data_file: os.PathLike, result_dir: os.PathLike, allow_cache = True):
    dataset = Dataset(data_file, allow_cache=allow_cache)
    eeg, labels = dataset.prepare_for_motion_intention_detection(config["data"]["interval"])
    ensure_dir(result_dir)
    
    # 初始化关联性矩阵
    initial_matrix_path = result_dir / path_conf["initial_matrix"]
    if os.path.exists(initial_matrix_path):
        matrix = np.loadtxt(initial_matrix_path, delimiter=",")
    else:
        matrix = SPMI_1epoch(eeg[0], 6, 2)
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

    # 构建模型
    model_conf = config["model"]
    gcn_net_model = GcnNet(
        node_embedding_dims=model_conf["node_embedding_dim"],
        class_num=model_conf["class_num"],
        adjacent_matrix=matrix
    )
    
    train_conf = config["train"]
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

    # 训练模型
    iter_num = train_conf["iteration_num"]
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
    
    # 保存训练统计数据
    with open(ensure_dir(result_dir / path_conf["training_stats"]), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss", "train_acc", "test_loss", "test_acc"])
        for i in range(len(result["train_loss"])):
            writer.writerow([
                result["train_loss"][i],
                result["train_acc"][i],
                result["test_loss"][i],
                result["test_acc"][i]
            ])
    
    # 画出acc和loss的曲线
    plt.figure()
    plt.plot(result["train_loss"], label="train_loss")
    plt.plot(result["test_loss"], label="test_loss")
    plt.legend()
    plt.savefig(ensure_dir(result_dir / path_conf["loss_plot"]))
    plt.figure()
    plt.plot(result["train_acc"], label="train_acc")
    plt.plot(result["test_acc"], label=f"test_acc")
    plt.legend()
    plt.savefig(ensure_dir(result_dir / path_conf["acc_plot"]))
    
    trained_matrix = gcn_net_model.get_matrix().cpu()
    
    # 保存训练后的关联性矩阵
    np.savetxt(ensure_dir(result_dir / path_conf["trained_matrix"]), trained_matrix, fmt="%.6f", delimiter=",")
    
    # 画出关联性矩阵
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
    
    # 保存模型
    torch.save(gcn_net_model, ensure_dir(result_dir / path_conf["model"]))
 
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
    run_parser = sub_parasers.add_parser("run", help="Run model")
    run_parser.add_argument("model_file", help="Path to model file")
    run_parser.add_argument("subject_indices", nargs="+", help="Indices of subjects whose data will be used for running")
    args = parser.parse_args()

    if args.command == "train":
        indices = [int(i) for i in args.subject_indices]
        data_files = {k: v for k, v in get_data_files().items() if k in indices}
        for [subj, data_file] in data_files.items():
            print(f"Training model using data from subject {subj}...")
            train(data_file, project_root / f"result/sub-{subj:02d}" / exp_name, not args.no_cache)
    if args.command == "run":
        indices = [int(i) for i in args.subject_indices]
        data_files = [v for k, v in get_data_files().items() if k in indices]
        run(args.model_file, data_files)