import numpy as np
import streamlit as st

from why import display as display
from why import interpret as interpret
from why import Explainer


def write(exp: Explainer):
    st.title("Local Effects")
    dataset = st.selectbox(
        "Select dataset on which to inspect local effects", ["Test", "Train"],
    )
    if dataset == "Train":
        X = exp.X_train
        preds = exp.train_preds
    else:
        X = exp.X_test
        preds = exp.test_preds
    st.markdown("#### Distribution of Model Predictions")
    distribution_plot = st.empty()
    p_min, p_max = st.slider(
        "Select range of predictions to explain",
        min_value=0.0,
        max_value=1.0,
        value=(0.9, 1.0),
        step=0.01,
    )
    fig, ax = display.plot_predictions(preds, p_min, p_max)
    distribution_plot.pyplot()

    selected_ids = np.where(np.logical_and(preds >= p_min, preds <= p_max))[0]

    local_effects_method = st.selectbox(
        "Select a local effects method",
        ["Don't display local effects", "Shapley values"],
    )
    if not local_effects_method == "Don't display local effects":
        n_feats = st.number_input(
            "Select the number of features to show", value=3, min_value=3, max_value=10,
        )
        st.markdown(f"#### Shapley Values for the Top {n_feats} Features")
        shap_values = interpret.shap_values(exp.model, X)
        feat_values, local_effects = interpret.shap_feature_values(
            exp.model, X, shap_values, selected_ids, n_feats=n_feats
        )
        st.table(display.format_local_explanations(feat_values))
