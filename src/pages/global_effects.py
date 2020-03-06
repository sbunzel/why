from typing import Any, Dict

import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

import streamlit as st


def write(session: Dict[str, Any]):
    st.title("Global Effects")
    st.markdown("#### Partial Dependence of Predictions on a Feature")
    feat = st.selectbox(
        "Please select a feature",
        ["Don't plot partial dependence"] + sorted(session["X_train"].columns),
    )
    if not feat == "Don't plot partial dependence":
        dataset = st.selectbox(
            "Please select dataset on which to calculate partial depedence",
            ["Test", "Train"],
        )
        if dataset == "Train":
            X = session["X_train"]
        else:
            X = session["X_valid"]
        plot_partial_dependence(
            session["m"],
            X,
            features=[feat],
            feature_names=X.columns,
            grid_resolution=20,
        )
        plt.tight_layout()
        st.pyplot()
