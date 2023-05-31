import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import xlwt


project_root = Path(__file__).parent.parent.parent


def get_data_files(data_dir: os.PathLike = project_root / "data"):
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    files: dict[int, str] = dict()
    for f in all_files:
        match = re.match(r"^ws_subj(\d+)\.npy$", f)
        if match:
            files[int(match.group(1))] = os.path.join(data_dir, f)
    return files


def load_config(task_name: str) -> Dict[str, Any]:
    config_file = project_root / f"config/{task_name}.json"
    return json.load(open(config_file, "r"))

def ensure_dir(path: os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

def save_to_sheet(
    wb: xlwt.Workbook, 
    sheet_name: str, 
    data: List[List], 
    headers: Optional[List[str]] = None):
    sheet = wb.add_sheet(sheet_name)
    if headers:
        assert len(headers) == len(data[0]), "headers must have the same length as data"
        for i, header in enumerate(headers):
            sheet.write(0, i, header)
    offset = 1 if headers else 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.write(i + offset, j, data[i][j])
    return sheet