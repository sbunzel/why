from typing import Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

__all__ = ["Explainer"]


class Explainer:
    """The core data container holding state shared across analyses.

    Args:
        train (pd.DataFrame): A DataFrame with training features and targets.
        test (pd.DataFrame): A DataFrame with test features and targets.
        target (str): The name of the target column.
        model (BaseEstimator): A Scikit-Learn model.
        features (Optional[Sequence[str]], optional): Column names of the features to use. If None (the default), use all columns except for the target column.
        mode (str, optional): The problem type. Supported options: "binary_classification". Defaults to "binary_classification".
        random_feature (bool, optional): Insert a random feature to test its effect on feature importance estimation. Defaults to True.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        model: BaseEstimator,
        features: Optional[Sequence[str]] = None,
        mode: str = "binary_classification",
        random_feature: bool = True,
    ):
        self.check_column_names(train, test)

        self.features = features if features else train.columns
        self.train = train[self.features].copy()
        self.test = test[self.features].copy()
        self.target = target
        self.model = model
        self.mode = mode
        self.random_feature = random_feature

        self.check_data_types()
        self.add_properties()

    def check_data_types(self) -> None:
        """Checks that the data types of training and test data are compatible and that all features are of numeric dtype to facilitate integration with scikit-learn."""
        diffs_dtypes = [
            {a: self.train[a].dtype, b: self.test[a].dtype}
            for a, b in zip(self.train.columns, self.test.columns)
            if self.train[a].dtype != self.test[b].dtype
        ]
        if diffs_dtypes:
            raise ValueError(
                f"Train and test are expected to have the same schema. Incompatible data types: {diffs_dtypes}."
            )
        non_numeric_cols = [
            {col: self.test[col].dtype}
            for col in set(self.features) - set([self.target])
            if not pd.api.types.is_numeric_dtype(self.test[col])
        ]
        if non_numeric_cols:
            raise ValueError(
                f"All feature columns are expected to be numeric dtype. Columns {non_numeric_cols} are not."
            )

    def add_properties(self):
        if self.random_feature:
            self.train = self.insert_random_num(self.train)
            self.test = self.insert_random_num(self.test)
        self.prepare_data()
        self.maybe_fit_model()
        self.get_predictions()
        self.feature_names = self.X_train.columns

    def prepare_data(self):
        """Splits the training and test data into feature matrix and target vector."""
        self.X_train, self.y_train = (
            self.train.drop(columns=self.target),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.test.drop(columns=self.target),
            self.test[self.target],
        )

    def maybe_fit_model(self) -> None:
        """Checks if the provided model is already fitted and fits it on the training data if not."""
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            self.model = self.model.fit(self.X_train, self.y_train)

    def get_predictions(self) -> None:
        """Derives predictions for the train and test data from the fitted model.

        Raises:
            NotImplementedError: Raise a error when a mode is requested that has not been implemented yet.
        """
        if self.mode == "binary_classification":
            self.train_preds = self.model.predict_proba(self.X_train)[:, 1]
            self.test_preds = self.model.predict_proba(self.X_test)[:, 1]
        elif self.mode == "multi_class_classification":
            self.train_preds = self.model.predict_proba(self.X_train)
            self.test_preds = self.model.predict_proba(self.X_test)
        else:
            raise NotImplementedError(
                f"Problem type {self.mode} has not been implemented yet."
            )

    def get_data_summary(self) -> str:
        """Summarizes the dimensions and target column of the input data.

        Returns:
            str: A string describing the input data.
        """
        n_rows = self.train.shape[0] + self.test.shape[0]
        n_cols = self.train.shape[1]
        return f"There are **{n_rows}** observations and **{n_cols - 1}** features in this dataset. The target variable is **{self.target}.**"

    def get_class_test_preds(self, class_name: Optional[str]) -> np.ndarray:
        if self.mode == "binary_classification":
            return self.test_preds
        else:
            return self.test_preds[:, np.where(self.model.classes_ == class_name)][
                :, 0
            ][:, 0]

    @staticmethod
    def check_column_names(train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Checks that train and test have the same columns.
        This implementation assumes that the column order is important. This assumption could be relaxed in the future.

        Args:
            train (pd.DataFrame): The training data provided.
            test (pd.DataFrame): The test data provided.

        Raises:
            ValueError: Raise if train and test have incompabitle shapes of column names.
        """
        if not train.shape[1] == test.shape[1]:
            raise ValueError(
                "Train and test are expected to have the same number of columns."
            )
        else:
            diffs_column_names = [
                (a, b) for a, b in zip(train.columns, test.columns) if a != b
            ]
            if diffs_column_names:
                raise ValueError(
                    f"Train and test are expected to have the same column names. Incompatible column names: {diffs_column_names}."
                )

    @staticmethod
    def insert_random_num(df: pd.DataFrame, copy=False) -> pd.DataFrame:
        """Adds a random numerical feature to the data.

        Args:
            df (pd.DataFrame): The input feature table.
            copy (bool, optional): If False, modify the input data in place to save memory. If True, create a copy. Defaults to False.

        Returns:
            pd.DataFrame: The input data with a random numerical feature added.
        """
        if copy:
            df = df.copy()
        df["RANDOM_NUM"] = np.random.rand(df.shape[0])
        return df
