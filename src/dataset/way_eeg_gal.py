import json
from typing import cast, List, Dict, Tuple, NamedTuple, Literal, Optional, Union

import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape
from mne import EpochsArray, create_info

from utils.common import project_root, dict_to_namedtuple


data_dir = project_root / "data"

class ElectrodeInfo(NamedTuple):
    group: int
    coordinate: Tuple[float, float, float]

class TrialMetadata(NamedTuple):
    led_on: float
    led_off: float
    trial_start: float
    trial_end: float
    surface: Literal["sandpaper", "silk", "suede"]
    weight: Literal[165, 330, 660]

class SeriesMetadata(NamedTuple):
    participant: int
    series: int
    channels: Dict[Literal["eeg", "emg", "kin"], List[str]]
    trials: list[TrialMetadata]

class Metadata:
    eeg_layout = {
        k: ElectrodeInfo(v["group"], tuple(v["coordinate"]))
        for k, v in json.load(open(data_dir / "eeg_layout.json", "r")).items()
    }
    
    @staticmethod
    def load(participant: int, series: int) -> SeriesMetadata:
        file = data_dir / f"sub-{participant:02d}/series-{series:02d}/metadata.json"
        if not file.exists():
            raise FileNotFoundError(f"Metadata file not found: {file}")
        metadata = json.load(open(file, "r"))
        return dict_to_namedtuple(metadata, SeriesMetadata)

EEG_TRIAL_TYPE = NDArray[Shape["32, *"], npt.Float64]  
EMG_TRIAL_TYPE = NDArray[Shape["5, *"], npt.Float64]  
KIN_TRIAL_TYPE = NDArray[Shape["45, *"], npt.Float64]

ALL_SURFACES = ["sandpaper", "silk", "suede"]
ALL_WEIGHTS = [165, 330, 660]
SAMPLING_RATES = dict(
    eeg=500,
    emg=4000,
    kin=500
)

class Dataset:
    def __init__(self, participant:int, series: int, load = True):
        self.participant = participant
        self.series = series
        self.filename = data_dir / f"sub-{participant:02d}/series-{series:02d}/samples.npz"
        if load:
            self.load()
        self._loaded = load
    
    def load(self):
        if self._loaded:
            return
        data: NDArray = np.load(self.filename, allow_pickle=True)
        self.eeg = cast(NDArray, data["eeg"])
        self.emg = cast(NDArray, data["emg"])
        self.kin = cast(NDArray, data["kin"])
        for i in range(len(self.eeg)):
            self.eeg[i] = self.eeg[i].transpose(1, 0)
            self.emg[i] = self.emg[i].transpose(1, 0)
            self.kin[i] = self.kin[i].transpose(1, 0)
        self._loaded = True
    
    def get_metadata(self) -> SeriesMetadata:
        if self._metadata is None:
            self._metadata = Metadata.load(self.participant, self.series)
        return self._metadata
    
    def pick_trial(self, trial: int) -> Tuple[EEG_TRIAL_TYPE, EMG_TRIAL_TYPE, KIN_TRIAL_TYPE]:
        assert 0 <= trial < len(self.eeg), "trial index out of range"
        return self.eeg[trial], self.emg[trial], self.kin[trial]
    
    def stack_trials(self, 
        type: Literal["eeg", "emg", "kin"],
        n_samples: Union[int, Literal["min", "max", "mean"]] = "min",
        trials: Optional[List[int]] = None) -> NDArray[Shape["*, *, *"], npt.Float64]:
        if trials is None:
            trials = list(range(len(self.eeg)))
        else:
            trials = list(set(trials))
            assert all([0 <= i < len(self.eeg) for i in trials]), "trial index out of range"
        if n_samples == "min":
            n_samples = min([self.eeg[i].shape[1] for i in trials])
        elif n_samples == "max":
            n_samples = max([self.eeg[i].shape[1] for i in trials])
        elif n_samples == "mean":
            n_samples = int(np.mean([self.eeg[i].shape[1] for i in trials]))
        data = getattr(self, type)
        return np.stack([
            data[i][:,:n_samples] if data[i].shape[1] >= n_samples 
                else np.pad(data[i], ((0, 0), (0, n_samples - data[i].shape[1])), "constant") 
            for i in trials
        ], axis=0)
    
    def to_epochs_array(self, 
        type: Literal["eeg", "emg", "kin"],
        n_samples: Union[int, Literal["min", "max", "mean"]] = "min",
        trials: Optional[List[int]] = None) -> EpochsArray:
        data = self.stack_trials(type, n_samples, trials)
        metadata = self.get_metadata()
        channel_type = "misc" if type == "kin" else type
        info = create_info(metadata.channels[type], SAMPLING_RATES[type], channel_type)
        SURFACE_MAP = { s: i for i, s in enumerate(ALL_SURFACES) }
        WEIGHT_MAP = { w: i for i, w in enumerate(ALL_WEIGHTS) }
        surfaces = [m.surface for m in metadata.trials]
        weights = [m.weight for m in metadata.trials]
        if all([s == surfaces[0] for s in surfaces]):
            event_id = { str(w): i for i, w in enumerate(ALL_WEIGHTS) }
            events = np.array([[idx, 0, SURFACE_MAP[str(t.weight)]] for idx, t in enumerate(metadata.trials)])
        elif all([w == weights[0] for w in weights]):
            event_id = { s: i for i, s in enumerate(ALL_SURFACES) }
            events = np.array([[idx, 0, WEIGHT_MAP[t.surface]] for idx, t in enumerate(metadata.trials)])
        else:
            event_id = { f"{s}|{w}": i * len(ALL_WEIGHTS) + j for i, s in enumerate(ALL_SURFACES) for j, w in enumerate(ALL_WEIGHTS) }
            events = np.array([[idx, 0, event_id[f"{t.surface}|{t.weight}"]] for idx, t in enumerate(metadata.trials)])
        return EpochsArray(data, info, events, event_id=event_id)