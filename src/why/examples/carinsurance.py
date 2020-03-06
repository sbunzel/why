from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils import read_config, get_root_dir


class CarInsurance:
    def __init__(self, config: Dict[str, Any], datapath: Path) -> None:
        """Initialize the car insurance data set

        Args:
            config (Dict[str, Any]): Configuration file for the car insurance data
            datapath (Path): Path to the folder containing the car insurance data
        """
        self.config = config
        self.datapath = datapath

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read the insurance data and apply the data transformation pipeline to train and test

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The prepared car insurance train and test set
        """
        train = self.read_data("carInsurance_train.csv").pipe(self.preparation_pipe)
        test = self.read_data("carInsurance_train.csv").pipe(self.preparation_pipe)
        return train, test

    def read_data(self, filename: str) -> pd.DataFrame:
        """Read the car insurance data into a Pandas DataFrame

        Returns:
            pd.DataFrame: The raw car insurance train or test set
        """
        date_parser = partial(pd.to_datetime, format="%H:%M:%S")
        return pd.read_csv(
            self.datapath / filename,
            parse_dates=["CallStart", "CallEnd"],
            date_parser=date_parser,
        ).set_index("Id")

    def preparation_pipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine preprocessing steps for the car insurance data in one function

        Args:
            df (pd.DataFrame): Raw car insurance data

        Returns:
            pd.DataFrame: Transformed car insurance data
        """
        return (
            df.pipe(self.add_call_time_features)
            .pipe(self.map_binary_cols)
            .rename(self.config["colmapper"], axis=1)
            .drop(["CallStart", "CallEnd"], axis=1)
        )

    def add_call_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add new features based on CallStart and CallEnd

        Args:
            df (pd.DataFrame): Car insurance data

        Returns:
            pd.DataFrame: Enhanced car insurance data
        """
        return df.assign(
            LastCallDurationSecs=lambda df: (
                df["CallEnd"] - df["CallStart"]
            ).dt.seconds,
            LastCallHourOfDay=lambda df: df["CallStart"].dt.hour,
        )

    def map_binary_cols(
        self, df: pd.DataFrame, cols: List[str] = ["CarLoan", "Default", "HHInsurance"]
    ) -> pd.DataFrame:
        """Map binary columns to more expressive values

        Args:
            df (pd.DataFrame): Car insurance data
            cols (List[str], optional): Column names of binary columns to transform.
                Defaults to ["CarLoan", "Default", "HHInsurance"].

        Returns:
            pd.DataFrame: Transformed car insurance data
        """
        df = df.copy()
        for c in cols:
            df[c] = df[c].map({0: "no", 1: "yes"})
        return df


class InsuranceTransformer(ColumnTransformer):
    def __init__(self, config: Dict[str, Any], df: pd.DataFrame) -> None:
        """Initialize the car insurance transformer.
        Extends Scikit-Learn's ColumnTransformer.

        Args:
            config (Dict[str, Any]): Configuration file for the car insurance data set
        """
        self.config = config
        self.df = df
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

    def split_train_valid(self):
        """Split the training data into train and valid"""
        X = self.df.drop(self.config["target"], axis=1)
        y = self.df[self.config["target"]]
        self.X_t, self.X_v, self.y_t, self.y_v = train_test_split(X, y, stratify=y)
        return self

    def transform_train_valid(self):
        """Apply the transformation pipeline to the splitted training data"""
        assert hasattr(
            self, "X_t"
        ), "Please call 'split' on your data before transforming."

        X_t = self.fit_transform(self.X_t)
        X_v = self.transform(self.X_v)

        cat_names, feature_names = self.get_feature_names()
        train_trans = (
            pd.DataFrame(X_t, columns=feature_names, index=self.y_t.index)
            .astype({feat: int for feat in cat_names})
            .assign(**{self.config["target"]: self.y_t})
        )
        valid_trans = (
            pd.DataFrame(X_v, columns=feature_names, index=self.y_v.index)
            .astype({feat: int for feat in cat_names})
            .assign(**{self.config["target"]: self.y_v})
        )

        self.train_trans, self.valid_trans = train_trans, valid_trans
        return self

    def get_feature_names(self) -> Tuple[np.ndarray]:
        """Extract the feature names of the transformed data from the transformer

        Returns:
            Tuple[np.ndarray]: Categorical feature names, all feature names
        """
        ohe = self.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names(input_features=self.config["catcols"])
        feature_names = np.r_[cat_feature_names, self.config["numcols"]]
        return cat_feature_names, feature_names

    def save_train_valid(self, savedir: Path):
        """Save the transformed data

        Args:
            savedir (Path): Path to save to
        """
        if not savedir.is_dir():
            savedir.mkdir()
        self.train_trans.to_csv(savedir / "train.csv", index=False)
        self.valid_trans.to_csv(savedir / "test.csv", index=False)


def main():
    config = read_config("car_insurance_cold_calls.json")
    datapath = get_root_dir() / "data"
    train, test = CarInsurance(config=config, datapath=datapath / "raw").prepare_data()
    InsuranceTransformer(
        config, train
    ).split_train_valid().transform_train_valid().save_train_valid(
        datapath / "processed" / "carinsurance"
    )


if __name__ == "__main__":
    main()
