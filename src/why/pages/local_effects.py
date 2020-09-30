import numpy as np
import pandas as pd
import streamlit as st

from why import display, interpret
from why import Explainer


def write(exp: Explainer):
    st.title("Local Effects")
    if exp.mode == "binary_classification":
        dataset = st.selectbox(
            "Select dataset on which to inspect local effects",
            ["test", "train"],
            format_func=lambda x: x.title(),
        )
        if dataset == "train":
            preds = exp.train_preds
        else:
            preds = exp.test_preds
        distribution_plot = st.empty()
        min_value = float(
            round(preds.max() - max((preds.max() - preds.min()) / 10, 0.05), 2)
        )
        max_value = float(round(preds.max(), 2))

        p_min, p_max = st.slider(
            "Select range of predictions to explain",
            min_value=0.0,
            max_value=1.0,
            value=(min_value, max_value),
            step=0.01,
        )
        pred_df = pd.DataFrame(data={"target": 1, "prediction": preds})
        chart = display.plot_predictions(df=pred_df, p_min=p_min, p_max=p_max)
        distribution_plot.altair_chart(chart, use_container_width=True)

        selected_ids = np.where(np.logical_and(preds >= p_min, preds <= p_max))[0]

        local_effects_method = st.selectbox(
            "Select a local effects method",
            ["Don't display local effects", "Shapley values"],
        )
        if not local_effects_method == "Don't display local effects":
            n_feats = st.number_input(
                "Select the number of features to show",
                value=3,
                min_value=3,
                max_value=10,
            )
            st.markdown(f"#### Shapley Values for Top {n_feats} Features")
            shap_features = interpret.ShapFeatures(exp=exp, dataset=dataset)
            feat_values = shap_features.calculate_shapley_values().combine_top_features(
                ids=selected_ids, n_feats=n_feats
            )
            st.table(
                display.style_local_explanations(
                    feat_values, min_pred=preds.min(), max_pred=preds.max()
                )
            )
    else:
        st.warning(f"Local effects are not implemented yet for mode '{exp.mode}'")
