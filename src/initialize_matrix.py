import sys
from typing import cast, Union

import numpy as np
from numpy.typing import NDArray

from dataset.way_eeg_gal import WayEegGalDataset
from connectivity.PMI import SPMI_1epoch
from utils.common import project_root, get_data_files, load_config


def initialize_matrix(data_path: str, out_path: Union[str, None] = None):
    dataset = WayEegGalDataset(data_path)
    eeg, _ = dataset.prepare_for_motion_intention_detection()
    first_trial = cast(NDArray, eeg[0])
    spmi = SPMI_1epoch(first_trial, 5, 1)
    for i in range(spmi.shape[0]):
        for j in range(0, i):
            spmi[i, j] = spmi[j, i]

    if out_path is not None:
        np.savetxt(out_path, spmi, delimiter=",")
        
    return spmi


if __name__ == '__main__':
    data_files = get_data_files()
    if len(sys.argv) > 1:
        indices = [int(i) for i in sys.argv[1:]]
        data_files = {k: v for k, v in data_files.items() if k in indices}
    task = "motion_intention_detection"
    path_config = load_config(task)["path"]
    for [i, f] in data_files.items():
        print(f"Processing subject {i}...")
        initialize_matrix(f, project_root/ f"result/sub-{i:02}" / task / path_config["initial_matrix"])
