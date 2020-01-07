from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

__all__ = ["InsuranceTransformer"]


class InsuranceTransformer(ColumnTransformer):
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the car insurance transformer.
        Extends Scikit-Learn's ColumnTransformer.
        
        Args:
            config (Dict[str, Any]): Configuration file for the car insurance data set
        """
        self.config = config
        self.cat_cols = config["catcols"]
        self.num_cols = config["numcols"]
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
        super().__init__(
            [("cat", cat_pipe, self.cat_cols), ("num", num_pipe, self.num_cols)],
            sparse_threshold=0,
        )

    def prepare_train_valid(self, df: pd.DataFrame) -> Tuple[np.ndarray]:
        """Split the training data into train and valid and apply the transformation pipeline
        
        Args:
            df (pd.DataFrame): The training data
        
        Returns:
            Tuple[np.ndarray]: X_train, X_valid, y_train, y_valid
        """
        X = df.drop(self.config["target"], axis=1)
        y = df[self.config["target"]]
        X_t, X_v, y_t, y_v = train_test_split(X, y, stratify=y)
        X_t = self.fit_transform(X_t)
        X_v = self.transform(X_v)
        self.X_t, self.X_v, self.y_t, self.y_v = X_t, X_v, y_t, y_v
        return X_t, X_v, y_t, y_v

    def save_transformed(self, savepath: Path) -> Dict[str, Path]:
        """Save the Scikit-Learn transformer and the transformed data
        
        Args:
            savepath (Path): Path to save to
        
        Returns:
            Dict[str, Path]: A dictionary containing the filepaths
        """
        assert hasattr(
            self, "X_t"
        ), "Please call 'prepare_train_valid' on your data before saving."
        train_valid_file = (savepath / "train_valid_data.npz").resolve()
        transformer_file = (savepath / "transformer.pkl").resolve()
        np.savez(
            train_valid_file,
            X_train=self.X_t,
            X_valid=self.X_v,
            y_train=self.y_t,
            y_valid=self.y_v,
        )
        with open(transformer_file, mode="wb") as f:
            pickle.dump(self, f)
        return {"train_valid": train_valid_file, "transformer": transformer_file}
