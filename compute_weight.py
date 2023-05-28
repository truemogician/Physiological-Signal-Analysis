from pathlib import Path
import sys

import xlwt
from preprocess_data import *
from connectivity.PMI import *
from utils import get_data_files

def compute_weight(data_path: str, out_path: str):
    train, _ = get_data_check_intend(data_path)
    for eeg, _ in train:
        break
    weight = []
    data = eeg[0].numpy()
    for i in range(4):
        epoch = data[:, i, :]
        temp = SPMI_1epoch(epoch, 5, 1)
        weight.append(temp)

    workbook = xlwt.Workbook()
    for i in range(len(weight)):
        sheet = workbook.add_sheet(f"sheet_{i}")
        data = list(weight[i])
        for row in range(len(data)):
            for col in range(len(data[0])):
                sheet.write(row, col, data[row][col])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    workbook.save(out_path)


if __name__ == '__main__':
    data_files = get_data_files()
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    for [i, f] in data_files.items():
        print(f"Processing {i}...")
        compute_weight(f, f"result/sub-{i:02}/eeg_initial_weight.xls")
