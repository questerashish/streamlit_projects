"""File IO helpers for the Streamlit desktop suite."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_table_csv(file: Path, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(file, encoding=encoding)


def load_table_parquet(file: Path) -> pd.DataFrame:
    return pd.read_parquet(file)


def save_dataframe(df: pd.DataFrame, path: Path, fmt: str = "csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "xlsx":
        df.to_excel(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return path


def save_json(data: dict | list, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path


def export_zip(files: Iterable[Path], dest: Path) -> Path:
    import zipfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, arcname=file.name)
    return dest
