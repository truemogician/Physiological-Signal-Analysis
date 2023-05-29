import os
import re
from pathlib import Path


project_root = Path(__file__).parent.parent.parent

def get_data_files(data_dir: os.PathLike = project_root / "data"):
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files: dict[int, str] = dict()
    for f in all_files:
        match = re.match(r"^ws_subj(\d+)\.npy$", f)
        if match:
            files[int(match.group(1))] = os.path.join(data_dir, f)
    return files
