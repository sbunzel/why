from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import inspection
import streamlit as st

from .explainer import Explainer


class PermutationImportance:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.X = {"train": self.exp.X_train, "test": self.exp.X_test}
        self.y = {"train": self.exp.y_train, "test": self.exp.y_test}

    def maybe_sample(
        self, X: pd.DataFrame, y: pd.Series, sample_size: int = 1000
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if X.shape[0] > sample_size:
            ids = np.random.choice(X.index, size=sample_size, replace=False)
        else:
            ids = X.index
        return X.loc[ids], y.loc[ids]

    def calculate_importance(
        self, dataset: str = "test", sample_size: int = 1000, **kwargs
    ):
        self.dataset = dataset
        X, y = self.maybe_sample(
            X=self.X[dataset], y=self.y[dataset], sample_size=sample_size
        )
        self.imp = inspection.permutation_importance(
            estimator=self.exp.model, X=X.values, y=y.values, **kwargs
        )
        return self

    def plot(self, top_n: int = 15):
        sorted_idx = self.imp.importances_mean.argsort()[-top_n:]
        top_n_imp = self.imp.importances[sorted_idx].T
        fig, ax = plt.subplots()
        ax.boxplot(top_n_imp, vert=False, labels=self.exp.feature_names[sorted_idx])
        ax.set(
            title=f"Permutation Importances (on the {self.dataset} set)",
            xlabel="Absolute Importance",
        )
        return fig


class ImpurityImportance:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp

    def calculate_importance(self, dataset: str = "train", **kwargs):
        if dataset == "test":
            st.info("Impurity Importance is only available on the training data.")
        self.imp = self.exp.model.feature_importances_
        return self

    def plot(self, top_n: int = 15):
        sorted_idx = self.imp.argsort()[-top_n:]
        feature_names = self.exp.feature_names[sorted_idx]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, self.imp[sorted_idx], align="center")
        ax.set(
            title="Impurity-based Importances (on the train set)",
            xlabel="Absolute Importance",
            yticks=y_pos,
            yticklabels=feature_names,
        )
        return fig


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return explainer.shap_values(X)


def shap_feature_values(
    model,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    ids: Union[np.ndarray, List[int]],
    n_feats: int = 5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    # TODO: Handle case when zero samples where selected gracefully
    top_feat_ids = np.argsort(np.abs(shap_values[1][ids, :]))[:, ::-1][:, :n_feats]
    sample_p1s = np.round(model.predict_proba(X.iloc[ids, :])[:, 1], 4)
    sample_shaps = np.array(
        [ar[feats] for ar, feats in zip(shap_values[1][ids], top_feat_ids)]
    )

    sample_feature_values = np.array(
        [
            [
                f"{c} = {v} ( {s:+.2f} )"
                for (c, v), s in zip(X.iloc[sample, feats].iteritems(), shaps)
            ]
            for sample, feats, shaps in zip(ids, top_feat_ids, sample_shaps)
        ]
    )
    top_n_feats = [f"Feature {i+1}" for i in range(n_feats)]
    feat_values = (
        pd.DataFrame(sample_feature_values, columns=top_n_feats)
        .assign(Prediction=sample_p1s)
        .sort_values("Prediction", ascending=False)
    )
    return feat_values, sample_shaps
