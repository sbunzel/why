from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import inspection
from sklearn.utils import Bunch
import streamlit as st

from .explainer import Explainer


class PermutationImportance:
    def __init__(self, exp: Explainer, dataset="test") -> None:
        self.exp = exp
        self.dataset = dataset
        if dataset == "train":
            self.X = exp.X_train.values
            self.y = exp.y_train
        elif dataset == "test":
            self.X = exp.X_test.values
            self.y = exp.y_test
        self.imp = self.calculate_importance()

    def calculate_importance(self) -> Bunch:
        imp = inspection.permutation_importance(
            estimator=self.exp.model, X=self.X, y=self.y, n_jobs=-1
        )
        return imp

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
    def __init__(self, exp: Explainer, dataset="test") -> None:
        self.exp = exp
        if dataset == "test":
            st.info("Impurity Importance is only available on the training data.")
        self.imp = self.calculate_importance()

    def calculate_importance(self) -> Bunch:
        imp = self.exp.model.feature_importances_
        imp = imp / len(imp)
        return imp

    def plot(self, top_n: int = 15):
        sorted_idx = self.imp.argsort()[-top_n:]
        feature_names = self.exp.feature_names[sorted_idx]
        fig, ax = plt.subplots()
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, self.imp[sorted_idx], align="center")
        ax.set(
            title="Impurity-based Importances (on the train set)",
            xlabel="Normalized Importance",
            yticks=y_pos,
            yticklabels=feature_names,
        )
        return fig


@st.cache()
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
