from typing import List, Iterable, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import inspection
import streamlit as st

from .explainer import Explainer


class PermutationImportance:
    def __init__(self, exp: Explainer, dataset="test", **kwargs) -> None:
        self.exp = exp
        self.dataset = dataset
        if dataset == "train":
            X = exp.X_train.values
            y = exp.y_train
        elif dataset == "test":
            X = exp.X_test.values
            y = exp.y_test
        self.imp = inspection.permutation_importance(
            estimator=exp.model, X=X, y=y, **kwargs
        )

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


# @st.cache
def feature_importance(type: str, feature_names: Iterable[str], **kwargs):
    m = kwargs["estimator"]
    if type == "impurity":
        imp = m.feature_importances_
        imp = imp / len(imp)
        sorted_idx = imp.argsort()[-15:]
        return imp[sorted_idx], feature_names[sorted_idx]
    elif type == "permutation":
        imp = inspection.permutation_importance(**kwargs)
        sorted_idx = imp.importances_mean.argsort()[-15:]
        return imp.importances[sorted_idx].T, feature_names[sorted_idx]
    else:
        raise NotImplementedError(
            f"Feature importance method {type} is not defined. Use either `impurity` or `permutation`."
        )


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
