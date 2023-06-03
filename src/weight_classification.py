import time
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import cast, List, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape
from matplotlib import pyplot as plt
import mne
import xlwt

from model.GcnNet import GcnNet
from model.utils import train_model, run_model
from dataset.way_eeg_gal import Dataset, Metadata, parse_indices, get_default_result_dir_name, SAMPLING_RATES, EMG_DATA_TYPE
from dataset.utils import create_data_loader, create_train_test_loader
from utils.common import load_config, ensure_dir, save_to_sheet, project_root
from utils.visualize import NodeMeta, PlotStyle, visualize_matrix
from algorithm.entropy import spmi


dataset_name = "WAY-EEG-GAL"
exp_name = Path(__file__).stem
config = load_config(exp_name)
path_conf = config["path"]

mne.set_log_level("WARNING")

def preprocess(
    dataset: Union[Dataset, Tuple[int, int]],
    sampling_rate = SAMPLING_RATES["emg"],
    cache = False) -> Tuple[EMG_DATA_TYPE, NDArray[Shape["*"], npt.Float64]]:
    assert SAMPLING_RATES["emg"] % sampling_rate == 0, f"sampling_rate must be a divisor of {SAMPLING_RATES['emg']}"
    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset[0], dataset[1], False)
    cache_filename = f"sub-{dataset.participant:02d}_series-{dataset.series:02d}(sampling_rate={sampling_rate}).npz"
    cache_file = project_root / "cache" / dataset_name / exp_name / cache_filename
    if cache and cache_file.exists():
        data = np.load(cache_file)
        return data["emg"], data["labels"]
    dataset.load()
    epochs_array = dataset.to_epochs_array("emg")
    start_time = time.time()
    # 对数据进行10-1000Hz的滤波
    epochs_array.filter(10, 1000, picks="emg")
    # 降采样
    if sampling_rate != SAMPLING_RATES["emg"]:
        epochs_array.resample(sampling_rate)
    emg = epochs_array.get_data()
    labels = np.array([e[2] for e in epochs_array.events])
    print(f"Preprocess Time: {time.time() - start_time:.2f}s")
    if cache:
        np.savez_compressed(ensure_dir(cache_file), emg=emg, labels=labels)
    return emg, labels

def train(
    data_index: Union[Tuple[int, int], List[Tuple[int, int]]],
    result_dir: os.PathLike, 
    cache = True,
    save_results = True,
    batch = 1):   
    data_indices = data_index if isinstance(data_index, list) else [data_index]
    eeg, labels = preprocess(data_indices[0], sampling_rate=config["data"]["sampling_rate"], cache=cache)
    matrix_trial = eeg[0]
    for index in data_indices[1:]:
        eeg_, labels_ = preprocess(index, sampling_rate=config["data"]["sampling_rate"], cache=cache)
        eeg = np.concatenate((eeg, eeg_), axis=0)
        labels = np.concatenate((labels, labels_), axis=0)
        matrix_trial = np.concatenate((matrix_trial, eeg_[0]), axis=1)
    
    # 初始化关联性矩阵
    initial_matrix_path = ensure_dir(result_dir) / path_conf["initial_matrix"]
    if cache and os.path.exists(initial_matrix_path):
        matrix = np.loadtxt(initial_matrix_path, delimiter=",")
    else:
        start_time = time.time()
        matrix = spmi(matrix_trial, 6, 2)
        matrix_mask = np.full(matrix.shape, True, dtype=bool)
        np.fill_diagonal(matrix_mask, False)
        min = matrix.min(initial=sys.float_info.max, where=matrix_mask)
        max = matrix.max(initial=sys.float_info.min, where=matrix_mask)
        rescaled_min, rescaled_max = min / 2, max / 2 + 0.5
        matrix = (matrix - min) / (max - min) * (rescaled_max - rescaled_min) + rescaled_min
        print(f"SPMI Time: {time.time() - start_time:.2f}s")
        if cache:
            np.savetxt(ensure_dir(initial_matrix_path), matrix, fmt="%.6f", delimiter=",")

    model_conf = config["model"]
    train_conf = config["train"]
    min_loss = sys.float_info.max
    max_acc = 0
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
        optimizer_conf = train_conf["optimizer"]
        optimizer_attr = {k: v for k, v in optimizer_conf.items() if k != "name"}
        optimizer = getattr(torch.optim, optimizer_conf["name"])(
            gcn_net_model.parameters(),
            **optimizer_attr
        )
        loss_func_conf = train_conf["loss_function"]
        loss_func_attr = {k: v for k, v in loss_func_conf.items() if k != "name"}
        loss_func = getattr(nn, loss_func_conf["name"])(*loss_func_attr)
        start_time = time.time()
        result = train_model(
            gcn_net_model, 
            train_iter, 
            optimizer, 
            lambda out, target: loss_func(out, target.long()), 
            iter_num, 
            test_iter
        )
        print(f"Training time: {time.time() - start_time:.2f}s")
        training_results.append(result)   
        
        if result["test_loss"][-1] < min_loss:
            min_loss = result["test_loss"][-1]
            min_loss_model_idx = idx
            min_loss_result = result
            
        if result["test_acc"][-1] > max_acc:
            max_acc = result["test_acc"][-1]
            best_model = gcn_net_model
            best_model_idx = idx
            best_result = result
    
    # 保存训练统计数据
    result_headers = ["train_loss", "train_acc", "test_loss", "test_acc"]
    if save_results:
        stats_workbook = xlwt.Workbook()
        for i in range(batch):
            sheet_name = f"model-{i}"
            if batch > 1:
                if i == best_model_idx:
                    sheet_name += " (max_acc)"
                if i == min_loss_model_idx:
                    sheet_name += " (min_loss)"
            matrix = np.stack([training_results[i][header] for header in result_headers])
            save_to_sheet(stats_workbook, sheet_name, matrix.T, result_headers)  
    if batch > 1:
        average_result = {
            header: [np.mean([r[header][i] for r in training_results]) for i in range(iter_num)]
            for header in result_headers
        }
        if save_results:
            matrix = np.stack([average_result[header] for header in result_headers])
            save_to_sheet(stats_workbook, "average", matrix.T, result_headers) 
        print(f"Max Acc: test_acc={best_result['test_acc'][-1]:.4f}, test_loss={best_result['test_loss'][-1]:.4f}")
        print(f"Min Loss: test_acc={min_loss_result['test_acc'][-1]:.4f}, test_loss={min_loss_result['test_loss'][-1]:.4f}")
        print(f"Average: test_acc={average_result['test_acc'][-1]:.4f}, test_loss={average_result['test_loss'][-1]:.4f}")      
    if not save_results:
        return
    stats_workbook.save(ensure_dir(result_dir / path_conf["training_stats"]))
    # 保存最佳模型的关联性矩阵
    trained_matrix = cast(NDArray[Shape["N, N"], npt.Float64], best_model.get_matrix().cpu().numpy())
    np.savetxt(ensure_dir(result_dir / path_conf["trained_matrix"]), trained_matrix, fmt="%.6f", delimiter=",") 
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
    # 最佳模型关联性矩阵可视化
    channel_names = Metadata.load(*data_indices[0]).channels["emg"]
    metadata = [NodeMeta(n) for n in channel_names]
    styles = config["plot"]["matrix"]["styles"]
    plot_style = PlotStyle(
        "node" in styles and styles["node"] or dict(),
        "edge" in styles and styles["edge"] or dict(),
        "axis" in styles and styles["axis"] or dict(),
        "font" in styles and styles["font"] or dict(),
        "layout" in styles and styles["layout"] or dict()
    )
    figure = visualize_matrix(trained_matrix,  metadata, style=plot_style)
    figure.write_html(ensure_dir(result_dir / path_conf["matrix_plot"]))
    # 保存最佳模型
    torch.save(best_model, ensure_dir(result_dir / path_conf["model"]))

def run(model_file: os.PathLike, data_index: Union[Tuple[int, int], List[Tuple[int, int]]]):
    model = cast(nn.Module, torch.load(model_file))
    data_indices = data_index if isinstance(data_index, list) else [data_index]
    loss_func_conf = config["train"]["loss_function"]
    loss_func_attr = {k: v for k, v in loss_func_conf.items() if k != "name"}
    loss_func = getattr(nn, loss_func_conf["name"])(*loss_func_attr)
    avg_loss, avg_acc = 0.0, 0.0
    for index in data_indices:
        data, labels = preprocess(index, interval=config["data"]["interval"], cache=True)
        loader = create_data_loader(data, labels, batch_size=config["train"]["batch_size"]) 
        loss, acc = run_model(model, loader, lambda out, target: loss_func(out, target.long()))
        avg_loss += loss
        avg_acc += acc
        print(f"{index} Loss: {loss:.4f}, Acc: {acc:.4f}")
    if len(data_indices) > 1:
        avg_loss /= len(data_indices)
        avg_acc /= len(data_indices)
        print(f"Average Loss: {avg_loss:.4f}, Average Acc: {avg_acc:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train or run model")
    sub_parasers = parser.add_subparsers(dest="command")
    train_parser = sub_parasers.add_parser("train", help="Train model")
    train_parser.add_argument("data_file_indices", nargs="+", help="Indices of data to be used for training. Format: <participant>-<series>, or <participant> for all series of a participant")
    train_parser.add_argument("--no-cache", action="store_true", help="Whether to use cache for data preprocessing and initial matrix")
    train_parser.add_argument("--no-save", action="store_true", help="Whether to save stats, plots and model")
    train_parser.add_argument("--batch", type=int, default="1", help="Number of times the model will be trained")
    train_parser.add_argument("--result_dir", help="Path to directory where results will be saved")
    run_parser = sub_parasers.add_parser("run", help="Run model")
    run_parser.add_argument("model_file", help="Path to model file")
    run_parser.add_argument("data_file_indices", nargs="+", help="Indices of data to be used for running. Format: <participant>-<series>, or <participant> for all series of a participant")
    args = parser.parse_args()

    indices = parse_indices(args.data_file_indices)
    if args.command == "train":
        default_result_dir = project_root / "result" / dataset_name / exp_name / get_default_result_dir_name(indices)
        train(indices, args.result_dir if args.result_dir else default_result_dir, not args.no_cache, not args.no_save, args.batch)
    if args.command == "run":
        run(args.model_file, indices)