import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .models import get_model_scores

__all__ = ["Explainer"]


class Explainer:
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        model,
        mode: str = "binary_classification",
        random_feature: bool = True,
    ):
        self.train = train
        self.test = test
        self.target = target
        self.model = model
        self.mode = mode

        if random_feature:
            self.train = self.insert_random_num(self.train)
            self.test = self.insert_random_num(self.test)
        self.prepare_data()
        self.maybe_fit_model()
        self.get_model_scores()

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

    def get_model_scores(self):
        self.train_preds, self.test_preds = get_model_scores(
            self.model, self.X_train, self.X_test, self.y_train, self.y_test
        )
