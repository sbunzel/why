import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def get_config(dataset: str) -> Dict[str, Any]:
    dataset = dataset.lower().replace(" ", "_")
    config_path = (
        Path(__file__).parent.parent / "resources" / f"{dataset}.json"
    ).resolve()
    with open(config_path, mode="r") as f:
        config = json.load(f)
    return config


def get_data_summary(train: pd.DataFrame, test: pd.DataFrame, target: str) -> str:
    n_rows = train.shape[0] + test.shape[0]
    n_cols = train.shape[1] + test.shape[1]
    return f"There are **{n_rows}** observations and **{n_cols - 1}** features in this dataset. The target variable is **{target}.**"
