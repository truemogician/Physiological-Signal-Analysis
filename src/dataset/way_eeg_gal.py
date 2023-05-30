from pathlib import Path
import json
from typing import Tuple, NamedTuple

import numpy as np
from numpy.typing import NDArray
import mne

from utils.common import project_root


mne.set_log_level("WARNING")

class ElectrodeMetadata(NamedTuple):
    group: int
    coordinate: Tuple[float, float, float]


class Dataset:
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
        for k, v in json.load(open(project_root / "data/eeg_electrodes.json", "r")).items()
    }
    
    eeg_sfreq = 500
    
    emg_sfreq = 4000
    
    def __init__(self, filename: str, load = True, allow_cache = True):
        self.filename = filename
        self.allow_cache = allow_cache
        if load:
            self.load()
            
    def load(self):
        data: NDArray = np.load(self.filename, allow_pickle=True)
        eeg_data = data.item()["eeg"].transpose(0, 2, 1)
        emg_data = data.item()["emg"].transpose(0, 2, 1)
        labels = np.array(data.item()["label"], dtype=np.int64)
        
        self.eeg_samples = eeg_data.shape[2]
        self.emg_samples = emg_data.shape[2]
        self.trial_num = len(labels)

        # 创建info结构
        info_eeg = mne.create_info(
            ch_names=Dataset.eeg_channel_names,
            ch_types="eeg",
            sfreq=Dataset.eeg_sfreq
        )
        info_emg = mne.create_info(
            ch_names=Dataset.emg_channel_names,
            ch_types="emg",
            sfreq=Dataset.emg_sfreq
        )
        
        # 创建事件
        events = np.array([[idx, 0, label] for idx, label in enumerate(labels)])
        event_id = dict(weight_165=1, weight_330=2, weight_660=4)
        
        self.eeg_epochs = mne.EpochsArray(eeg_data, info_eeg, events, 0, event_id)
        self.emg_epochs = mne.EpochsArray(emg_data, info_emg, events, 0, event_id)
        self.labels = labels
        
    def prepare_for_motion_intention_detection(self, interval = 1000) -> Tuple[NDArray, NDArray]:
        '''
        构建用于GCN_LSTM实现EEG运动意图检测的数据。
        具体而言，因为2s时LED灯闪烁开始运动，因此取前4s的脑电数据，其中前2s为静息状态0，后2s为运动状态1。
        输出数据维度为(2x, 32, 1000)，其中x为样本数。
        '''
        assert 1000 % interval == 0, "interval must be a divisor of 1000" 
        use_cache = self.allow_cache and interval == 1000
        cache_file = Path(self.filename).parent / f"{Path(self.filename).stem}-motion_intention_detection.npz"
        if use_cache and cache_file.exists():
            data = np.load(cache_file)
            eeg, labels = data["eeg"], data["labels"]
        else:
            # 对数据进行0.05~50的滤波
            self.eeg_epochs.filter(0.05, h_freq=50)
            eeg = self.eeg_epochs.get_data()
            eeg = eeg[:, :, :2000] # 0-2s为静息状态，而运动在4-8s之间停止，因此取前4s数据
            
            # 将数据重新组合
            eeg = np.split(eeg, 2000 // interval, axis=2)
            eeg = np.concatenate(eeg, axis=0)
            
            # 构建运动意图标签
            new_trial_num = eeg.shape[0]
            labels = np.ones(new_trial_num)
            labels[:self.trial_num * 1000 // interval] = 0
            
            if use_cache:
                np.savez(cache_file, eeg=eeg, labels=labels)
        
        return eeg, labels
    
    def prepare_for_weight_classification(self):
        cache_file = Path(self.filename).parent / f"{Path(self.filename).stem}-weight_classification.npz"
        if self.allow_cache and cache_file.exists():
            data = np.load(cache_file)
            emg, labels = data["emg"], data["labels"]
        else:
            # 对数据进行10-1000Hz的滤波
            self.emg_epochs.filter(10, 1000)
            
            # 降采样至原采样率的1/4以缩小数据规模
            self.emg_epochs.resample(Dataset.emg_sfreq / 4)
            
            emg = self.emg_epochs.get_data()
            labels = self.labels           
            if self.allow_cache:
                np.savez(cache_file, emg=emg, labels=labels)
        
        return emg, labels