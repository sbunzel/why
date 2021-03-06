from pathlib import Path
from typing import Optional, Union, Tuple

import pandas as pd

__all__ = ["load_data"]


def load_data(
    dataset: str, data_home: Optional[Union[Path, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load prepared train and test datasets from the why GitHub repo.

    Args:
        dataset (str): The name of the dataset to load.
        data_home (Optional[Union[Path, str]], optional): The download and cache folder for the datasets. By default all data is store in `~/why_data` subfolders.

    Raises:
        NotImplementedError: Raise when a dataset is requested that is not available in the repo.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]: A tuple of the train and test data as pd.DataFrames and the name of the target column.
    """
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
    """Creates a path to cache data in, either in the standard directory ~/why_data or in the provided directory.

    Args:
        dataset (str): Name of the dataset to create a data path for.
        data_home (Optional[Union[Path, str]], optional): Home directory to create data paths in. Defaults to None.

    Returns:
        Path: The path to save cached data to.
    """
    return (
        Path(data_home) / dataset if data_home else Path.home() / "why_data" / dataset
    )


def _read_train_test(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads cached train and test data.

    Args:
        data_path (Path): Path to folder to read the data from.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of train_df and test_df.
    """
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    return train, test


def _write_train_test(data_path: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Writes the train and test data to the cache directory.

    Args:
        data_path (Path): Path to use for caching.
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame): Testing DataFrame.
    """
    if not data_path.parent.is_dir():
        data_path.parent.mkdir()
    data_path.mkdir(exist_ok=True)
    train.to_csv(data_path / "train.csv", index=False)
    test.to_csv(data_path / "test.csv", index=False)
