from pathlib import Path
import sys
from typing import cast, List, Union

import xlwt
from torch import Tensor
from numpy.typing import NDArray

from preprocess_data import *
from connectivity.PMI import SPMI_1epoch
from utils.common import get_data_files

def initialize_matrix(data_path: str, out_path: Union[str, None] = None, trial_num = 5):
    train, _ = get_data_motion_intention(data_path)
    eeg = cast(Tensor, train.dataset.tensors[0])
    matrices: List[NDArray] = []
    for i in range(min(trial_num, eeg.shape[0])):
        data = cast(NDArray, eeg[i].numpy())
        pmi = SPMI_1epoch(data, 5, 1)
        for i in range(pmi.shape[0]):
            for j in range(0, i):
                pmi[i, j] = pmi[j, i]
        matrices.append(pmi)

    if out_path is not None:
        workbook = xlwt.Workbook()
        for i in range(len(matrices)):
            sheet = workbook.add_sheet(f"trial_{i}")
            data = list(matrices[i])
            for row in range(len(data)):
                for col in range(len(data[0])):
                    sheet.write(row, col, data[row][col])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        workbook.save(out_path)
        
    return matrices


if __name__ == '__main__':
    data_files = get_data_files()
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    for [i, f] in data_files.items():
        print(f"Processing subject {i}...")
        initialize_matrix(f, f"result/sub-{i:02}/initial_matrix.xlsx")
