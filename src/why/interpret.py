from typing import List, Iterable, Union, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn import inspection
import streamlit as st


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
        raise NotImplementedError(f"Feature importance method {type} is not defined. Use either `impurity` or `permutation`.")


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
