import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
import streamlit as st

from why.explainer import Explainer


def write(exp: Explainer):
    st.title("Global Effects")
    st.markdown("#### Partial Dependence of Predictions on a Feature")
    feat = st.selectbox(
        "Please select a feature",
        ["Don't plot partial dependence"] + sorted(exp.X_train.columns),
    )
    if not feat == "Don't plot partial dependence":
        dataset = st.selectbox(
            "Please select dataset on which to calculate partial depedence",
            ["Test", "Train"],
        )
        if dataset == "Train":
            X = exp.X_train
        else:
            X = exp.X_test
        plot_partial_dependence(
            exp.model, X, features=[feat], feature_names=X.columns, grid_resolution=20,
        )
        plt.tight_layout()
        st.pyplot()
