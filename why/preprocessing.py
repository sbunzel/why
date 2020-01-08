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

__all__ = ["InsuranceTransformer", "remove_high_corrs", "get_high_corrs"]


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

        cat_names, feature_names = self.get_feature_names()
        X_t = pd.DataFrame(X_t, columns=feature_names).astype(
            {feat: int for feat in cat_names}
        )
        X_v = pd.DataFrame(X_v, columns=feature_names).astype(
            {feat: int for feat in cat_names}
        )

        self.X_t, self.X_v, self.y_t, self.y_v = X_t, X_v, y_t, y_v
        return X_t, X_v, y_t, y_v

    def get_feature_names(self) -> Tuple[np.ndarray]:
        """Extract the feature names of the transformed data from the transformer

        Returns:
            Tuple[np.ndarray]: Categorical feature names, all feature names
        """
        ohe = self.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names(input_features=self.config["catcols"])
        feature_names = np.r_[cat_feature_names, self.config["numcols"]]
        return cat_feature_names, feature_names

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
        train_valid_folder = (savepath / "train_valid_data").resolve()
        if not train_valid_folder.is_dir():
            train_valid_folder.mkdir(parents=True)
        transformer_file = (savepath / "transformer.pkl").resolve()
        if not transformer_file.parent.is_dir():
            transformer_file.parent.mkdir(parents=True)
        self.X_t.to_feather(train_valid_folder / "X_train.feather")
        self.X_v.to_feather(train_valid_folder / "X_valid.feather")
        np.savez(train_valid_folder / "targets.npz", y_train=self.y_t, y_valid=self.y_v)
        with open(transformer_file, mode="wb") as f:
            pickle.dump(self, f)
        return {"train_valid": train_valid_folder, "transformer": transformer_file}


def remove_high_corrs(df: pd.DataFrame, min_corr: float = 1.0) -> pd.DataFrame:
    """Drop highly correlated features from a DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing the features
        min_corr (float, optional): Minimum value for Pearson correlation to consider a correlation as high. Defaults to 1.0. . Defaults to 1.0.

    Returns:
        pd.DataFrame: The feature data with highly correlated features removed
    """
    high_corrs = get_high_corrs(df, min_corr)
    return df.drop(columns=high_corrs["feature1"].unique())


def get_high_corrs(df: pd.DataFrame, min_corr: float = 1.0) -> pd.DataFrame:
    """Find features with high correlation

    Args:
        df (pd.DataFrame): DataFrame containing the features
        min_corr (float, optional): Minimum value for Pearson correlation to consider a correlation as high. Defaults to 1.0.

    Returns:
        pd.DataFrame: The highly correlated features
    """
    corrs = df.corr().reset_index()
    return (
        pd.melt(corrs, id_vars=["index"])
        .dropna()
        .rename({"index": "feature1", "variable": "feature2", "value": "corr"}, axis=1)
        .query(f"feature1 != feature2 and abs(corr) >= {min_corr}")
        .sort_values(["corr", "feature1", "feature2"])
        .loc[::2]
        .reset_index(drop=True)
    )
