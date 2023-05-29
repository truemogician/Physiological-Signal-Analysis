from pathlib import Path
import json
from typing import Tuple, NamedTuple

import numpy as np
from numpy.typing import NDArray
import mne


class ElectrodeMetadata(NamedTuple):
    group: int
    coordinate: Tuple[float, float, float]


class WayEegGalDataset:
    eeg_channel_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz",
        "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7",
        "C3", "Cz", "C4", "T8", "TP9", "CP5",
        "CP1", "CP2", "CP6", "TP10", "P7", "P3",
        "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]
    
    # 1-三角前肌，2-肱桡肌，3-指屈肌，4-指总伸肌，5-骨间背肌
    emg_channel_names = ["EMG1", "EMG2", "EMG3", "EMG4", "EMG5"]
    
    eeg_electrode_metadata = {
        k: ElectrodeMetadata(v["group"], tuple(v["coordinate"]))
        for k, v in json.load(open(Path(__file__).parent.parent / "data/eeg_electrodes.json", "r")).items()
    }
    
    def __init__(self, filename: str, load = True, allow_cache = True):
        self.filename = filename
        self.allow_cache = allow_cache
        if load:
            self.load()
            
    def load(self):
        data: NDArray = np.load(self.filename, allow_pickle=True)
        eeg_data = data.item()["eeg"].transpose(0, 2, 1)
        emg_data = data.item()["emg"].transpose(0, 2, 1)
        label = np.array(data.item()["label"], dtype=int)

        # 构建事件
        sfreq_eeg = 500
        sfreq_emg = 4000

        # 创建info结构
        info_eeg = mne.create_info(
            ch_names=WayEegGalDataset.eeg_channel_names,
            ch_types="eeg",
            sfreq=sfreq_eeg
        )
        info_emg = mne.create_info(
            ch_names=WayEegGalDataset.emg_channel_names,
            ch_types="emg",
            sfreq=sfreq_emg
        )
        
        # 创建事件
        events = np.array([[idx, 0, label] for idx, label in enumerate(label)])
        event_id = dict(weight_165=1, weight_330=2, weight_660=4)
        
        self.eeg_epochs = mne.EpochsArray(eeg_data, info_eeg, events, 0, event_id)
        self.emg_epochs = mne.EpochsArray(emg_data, info_emg, events, 0, event_id)
        self.labels = np.array((label+0.1)/2, dtype=np.int64)
        
    def prepare_for_motion_intention_detection(self) -> Tuple[NDArray, NDArray]:
        '''
        构建用于GCN_LSTM实现EEG运动意图检测的数据。
        具体而言，因为2s时LED灯闪烁开始运动，因此取前4s的脑电数据，其中前2s为静息状态0，后2s为运动状态1。
        输出数据维度为(2x, 32, 1000)，其中x为样本数。
        '''
        cache_file = Path(self.filename).parent / f"{Path(self.filename).stem}-motion_intention.npz"
        if self.allow_cache and cache_file.exists():
            data = np.load(cache_file)
            eeg, labels = data["eeg"], data["labels"]
        else:
            # 对数据进行0.05~50的滤波
            self.eeg_epochs.filter(0.05, h_freq=50)
            eeg_data = self.eeg_epochs.get_data()[:, :, :2000]
            
            # 将数据拆分开
            eeg = np.concatenate((eeg_data[:, :, :1000], eeg_data[:, :, 1000:]), axis=0)
            
            # 构建运动意图标签
            trial_num = eeg.shape[0]
            labels = np.zeros((trial_num))
            labels[trial_num // 2:] = 1
            
            if self.allow_cache:
                np.savez(cache_file, eeg=eeg, labels=labels)
        
        return eeg, labels