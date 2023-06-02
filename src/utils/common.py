import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, NamedTuple

import xlwt


project_root = Path(__file__).parent.parent.parent

def load_config(task_name: str) -> Dict[str, Any]:
    config_file = project_root / f"config/{task_name}.json"
    return json.load(open(config_file, "r"))

def ensure_dir(path: os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path

T = TypeVar("T", bound=NamedTuple)
def dict_to_namedtuple(data: Dict[str, object], nt_class: Type[T]) -> T:
    fields = nt_class._fields
    field_types = nt_class.__annotations__
    values = []
    for field in fields:
        if field in data:
            value = data[field]
            if isinstance(value, dict) and field in field_types:
                nested_nt = dict_to_namedtuple(value, field_types[field])
                values.append(nested_nt)
            else:
                values.append(value)
        else:
            values.append(None)
    return nt_class(*values) # type: ignore

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