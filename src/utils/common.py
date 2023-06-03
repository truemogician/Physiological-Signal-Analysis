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

def is_namedtuple(tp: Type):
    if not (isinstance(tp, type) and issubclass(tp, tuple)):
        return False
    if not (hasattr(tp, "__annotations__") and hasattr(tp, "_fields") and hasattr(tp, "_field_defaults")):
        return False
    if not (isinstance(tp.__annotations__, dict) and isinstance(tp._fields, tuple) and isinstance(tp._field_defaults, dict)):
        return False
    return set(tp.__annotations__.keys()) == set(tp._fields)

T = TypeVar("T", bound=NamedTuple)
def dict_to_namedtuple(data: Dict[str, object], nt_class: Type[T]) -> T:
    def _convert_dict(value: Dict[str, Any], target_type: Type):
        if is_namedtuple(target_type):
            return dict_to_namedtuple(value, target_type)
        elif hasattr(target_type, "__origin__") and target_type.__origin__ is dict:
            args = target_type.__args__
            assert len(args) == 2
            value_type = args[1]
            if is_namedtuple(value_type):
                return {k: dict_to_namedtuple(v, value_type) for k, v in value.items()}
            elif hasattr(value_type, "__origin__"):
                origin = value_type.__origin__
                if origin is dict:
                    return {k: _convert_dict(v, value_type) for k, v in value.items()}
                elif origin is list:
                    return {k: _convert_list(v, value_type) for k, v in value.items()}
        return value
    
    def _convert_list(value: List[Any], target_type: Type[List]):
        assert hasattr(target_type, "__origin__") and target_type.__origin__ is list
        args = target_type.__args__
        assert len(args) == 1
        value_type = args[0]
        if is_namedtuple(value_type):
            return [dict_to_namedtuple(v, value_type) for v in value]
        elif hasattr(value_type, "__origin__"):
            origin = value_type.__origin__
            if origin is dict:
                return [_convert_dict(v, value_type) for v in value]
            elif origin is list:
                return [_convert_list(v, value_type) for v in value]
        return value
    
    fields = nt_class._fields
    field_types = nt_class.__annotations__
    values = []
    for field in fields:
        new_value = None
        if field in data and field in field_types:
            new_value = value = data[field]
            field_type = field_types[field]
            if isinstance(value, dict):
                new_value = _convert_dict(value, field_type)
            elif isinstance(value, list):
                new_value = _convert_list(value, field_type)
        values.append(new_value)
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