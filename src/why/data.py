from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd

__all__ = ["load_data"]


def load_data(
    dataset: str, data_home: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    TARGET_COLS = {
        "Car Insurance Cold Calls": "CarInsurance",
        "Cervical Cancer": "Biopsy",
    }
    dataset_folder = dataset.lower().replace(" ", "-")
    data_path = _build_data_path(dataset=dataset_folder, data_home=data_home)
    if not ((data_path / "train.csv").is_file() and (data_path / "test.csv").is_file()):
        if dataset in list(TARGET_COLS.keys()):
            train = pd.read_csv(
                f"https://raw.githubusercontent.com/sbunzel/why/master/data/{dataset_folder}/train.csv"
            )
            test = pd.read_csv(
                f"https://raw.githubusercontent.com/sbunzel/why/master/data/{dataset_folder}/test.csv"
            )
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not implemented.")
        _write_train_test(data_path, train, test)
    else:
        train, test = _read_train_test(data_path)
    return train, test, TARGET_COLS[dataset]


def _build_data_path(
    dataset: str, data_home: Optional[Union[Path, str]] = None
) -> Path:
    return (
        Path(data_home) / dataset if data_home else Path.home() / "why_data" / dataset
    )


def _read_train_test(data_path: Path):
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    return train, test


def _write_train_test(data_path: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
    if not data_path.parent.is_dir():
        data_path.parent.mkdir()
    data_path.mkdir(exist_ok=True)
    train.to_csv(data_path / "train.csv", index=False)
    test.to_csv(data_path / "test.csv", index=False)
