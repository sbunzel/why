from typing import Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

__all__ = ["Explainer"]


class Explainer:
    """The core data container holding state shared across analyses."""

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        model,
        features: Optional[Sequence[str]] = None,
        mode: str = "binary_classification",
        random_feature: bool = True,
    ):
        self.features = features if features else train.columns
        self.train = train[self.features].copy()
        self.test = test[self.features].copy()
        self.target = target
        self.model = model
        self.mode = mode

        if random_feature:
            self.train = self.insert_random_num(self.train)
            self.test = self.insert_random_num(self.test)
        self.prepare_data()
        self.maybe_fit_model()
        self.get_predictions()
        self.feature_names = self.train.columns

    def insert_random_num(self, df: pd.DataFrame) -> pd.DataFrame:
        df["RANDOM_NUM"] = np.random.rand(df.shape[0])
        return df

    def prepare_data(self) -> None:
        self.X_train, self.y_train = (
            self.train.drop(columns=self.target),
            self.train[self.target],
        )
        self.X_test, self.y_test = (
            self.test.drop(columns=self.target),
            self.test[self.target],
        )

    def maybe_fit_model(self) -> None:
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            self.model = self.model.fit(self.X_train, self.y_train)

    def get_predictions(self) -> None:
        if self.mode == "binary_classification":
            self.train_preds = self.model.predict_proba(self.X_train)[:, 1]
            self.test_preds = self.model.predict_proba(self.X_test)[:, 1]
        else:
            raise NotImplementedError(
                f"Problem type {self.mode} has not been implemented yet."
            )

    def get_data_summary(self) -> str:
        n_rows = self.train.shape[0] + self.test.shape[0]
        n_cols = self.train.shape[1]
        return f"There are **{n_rows}** observations and **{n_cols - 1}** features in this dataset. The target variable is **{self.target}.**"
