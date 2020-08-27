from typing import Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import shap
from sklearn import inspection
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from stratx import partdep
import streamlit as st

from .explainer import Explainer

__all__ = [
    "PermutationImportance",
    "ImpurityImportance",
    "ShapFeatures",
    "FeatureCorrelation",
]


class PermutationImportance:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.X = {"train": self.exp.X_train, "test": self.exp.X_test}
        self.y = {"train": self.exp.y_train, "test": self.exp.y_test}

    def calculate_importance(self, dataset: str = "test", **kwargs):
        self.dataset = dataset
        self.imp = self._permutation_importance(
            estimator=self.exp.model, X=self.X[dataset], y=self.y[dataset], **kwargs
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

    @staticmethod
    # TODO: Look into how to cache this calculation
    def _permutation_importance(
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.DataFrame,
        scoring: str,
        sample_size: int,
    ) -> Bunch:
        X = _maybe_sample(input=X, sample_size=sample_size).values
        y = _maybe_sample(input=y, sample_size=sample_size).values
        return inspection.permutation_importance(
            estimator=estimator, X=X, y=y, scoring=scoring, n_jobs=-1
        )


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


class ShapFeatures:
    def __init__(self, exp: Explainer, dataset: str) -> None:
        self.exp = exp
        self.dataset = dataset
        if self.dataset == "train":
            self.X = self.exp.X_train
        else:
            self.X = self.exp.X_test

    def calculate_shapley_values(self):
        # background_sample = _maybe_sample(self.exp.X_test, sample_size=200)
        explainer = shap.TreeExplainer(
            model=self.exp.model, feature_perturbation="tree_path_dependent"
        )
        self.shaps = explainer.shap_values(self.X)
        return self

    def combine_top_features(self, ids: np.ndarray, n_feats: int = 5) -> pd.DataFrame:
        if len(ids) > 0:
            top_feat_ids = np.argsort(np.abs(self.shaps[1][ids, :]))[:, ::-1][
                :, :n_feats
            ]
            sample_p1s = np.round(
                self.exp.model.predict_proba(self.X.iloc[ids, :])[:, 1], 4
            )
            sample_shaps = np.array(
                [ar[feats] for ar, feats in zip(self.shaps[1][ids], top_feat_ids)]
            )

            sample_feature_values = np.array(
                [
                    [
                        f"{c} = {v} ( {s:+.2f} )"
                        for (c, v), s in zip(
                            self.X.iloc[sample, feats].iteritems(), shaps
                        )
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
        else:
            feat_values = pd.DataFrame(columns=["Prediction"])
        return feat_values


class FeatureCorrelation:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp

    # TODO: Look into how to cache this calculation
    def calculate_correlation(self, method: str = "spearman", sample_size: int = 1000):
        X = _maybe_sample(self.exp.X_train, sample_size)
        self.corr = X.corr(method=method)
        return self

    def plot_dendrogram(self) -> Figure:
        corr_linkage = hierarchy.ward(self.corr)
        fig, ax = plt.subplots()
        _ = hierarchy.dendrogram(
            Z=corr_linkage,
            orientation="right",
            labels=list(self.exp.feature_names),
            ax=ax,
        )
        ax.set(
            title="Dendrogram of Feature Correlation Clusters", xlabel="Distance",
        )
        return fig


class ModelDependentPD:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp

    def plot(self, dataset: str, feature: str) -> Figure:
        if dataset == "train":
            X = self.exp.X_train
            # y = self.exp.y_train
        else:
            X = self.exp.X_test
            # y = self.exp.y_test
        disp = inspection.plot_partial_dependence(
            self.exp.model,
            X,
            features=[feature],
            feature_names=X.columns,
            grid_resolution=20,
            n_jobs=-1,
        )
        disp.axes_[0, 0].set(title="Model-dependent Partial Dependence", ylim=(0, 1))
        return disp.figure_


class ModelIndependentPD:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp

    def plot(self, dataset: str, feature: str) -> Figure:
        if dataset == "train":
            X = self.exp.X_train
            y = self.exp.y_train
        else:
            X = self.exp.X_test
            y = self.exp.y_test
        pdpx, pdpy, _ = partdep.plot_stratpd(X, y, feature, self.exp.target)
        fig, ax = plt.subplots()
        ax.plot(pdpx, pdpy)
        ax.set(
            title="Model-independent Partial Dependence (stratx)",
            xlabel=feature,
            ylabel="Partial dependence",
            ylim=(0, 1),
        )
        return fig


def _maybe_sample(
    input: Union[pd.DataFrame, pd.Series], sample_size: int = 1000
) -> Union[pd.DataFrame, pd.Series]:
    if input.shape[0] > sample_size:
        ids = np.random.choice(input.index, size=sample_size, replace=False,)
    else:
        ids = input.index
    return input.loc[ids]
