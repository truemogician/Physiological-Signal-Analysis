import json
import os
from argparse import ArgumentParser
from pathlib import Path

import scipy.io as sio
import numpy as np


def repack_original_windowed_data(data_file: os.PathLike, out_dir: os.PathLike, compress = True):
    """
    Read the original data file in MatLab format and repack it into a json metadata file and a npz samples file.
    
    The dimension of the samples file is (n_trials, n_samples, n_channels). But since different trials have different number of samples,
    each entry is actually a 1D object-type NDArray, whose elements are 2D arrays of shape (n_samples, n_channels).
    """
    ws = sio.loadmat(data_file, squeeze_me=True, struct_as_record=False)["ws"]
    metadata = dict(
        participant=ws.participant,
        series=ws.series,
        channels=dict(
            eeg=ws.names.eeg.tolist(),
            emg=ws.names.emg.tolist(),
            kin=ws.names.kin.tolist(),
        )
    )
    data = ws.win
    metadata["trials"] = [dict(
        led_on=t.LEDon,
        led_off=t.LEDoff,
        trial_start=t.trial_start_time,
        trial_end=t.trial_end_time,
        surface=t.surf_id,
        weight=int(t.weight_id[:3])
    ) for t in data]
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent="\t")
    eeg_data = np.array([t.eeg for t in data], dtype=object)
    emg_data = np.array([t.emg for t in data], dtype=object)
    kin_data = np.array([t.kin for t in data], dtype=object)
    if compress:
        np.savez_compressed(f"{out_dir}/samples.npz", eeg=eeg_data, emg=emg_data, kin=kin_data)
    else:
        np.savez(f"{out_dir}/samples.npz", eeg=eeg_data, emg=emg_data, kin=kin_data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_file", type=str, help="Path to the original data file in MatLab format")
    parser.add_argument("out_dir", type=str, help="Path to the output directory")
    parser.add_argument("--no-compress", action="store_false", dest="compress", help="Do not compress output file")
    args = parser.parse_args()
    repack_original_windowed_data(args.data_file, args.out_dir, args.compress)