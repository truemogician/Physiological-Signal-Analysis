import re
import os
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

sys.path.append(f"{Path(__file__).parent.parent}/src")
from dump_windowed_data import main # pylint: disable=import-error

def extract(zip_file: os.PathLike, out_dir: os.PathLike, delete = True):
    print(f"Extracting {zip_file} to {out_dir}...")
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(out_dir)
    if delete:
        os.remove(zip_file)
    print(f"Done {zip_file}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, 
        help="Path to the folder containing the zipped data files directly downloaded from figshare.")
    parser.add_argument("--no-delete", action="store_false", dest="delete",
        help="Do not delete the zipped data files after extraction.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist.")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a directory.")
    data_files = glob(f"{data_dir}/P*.zip")
    if len(data_files) == 0:
        raise FileNotFoundError(f"No data files found in {data_dir}.")
    pattern = re.compile(r".*P(\d+).zip$")
    extraction_task_args: Tuple[List[str], List[Path]] = ([], [])
    for data_file in data_files:
        match = pattern.match(data_file)
        assert match is not None, f"Illegal file name: {data_file}"
        participant = int(match.group(1))
        extraction_task_args[0].append(data_file)
        extraction_task_args[1].append(data_dir / f"sub-{participant:02d}/raw")
    
    with ThreadPoolExecutor() as executor:
        executor.map(extract, *extraction_task_args, [args.delete] * len(data_files))
    with ThreadPoolExecutor() as executor:
        executor.map(main, [
            [str(raw_dir), str(raw_dir.parent)]
            for raw_dir in extraction_task_args[1]
        ])