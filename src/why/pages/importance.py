import streamlit as st

import matplotlib.pyplot as plt

from why import display as display
from why import interpret as interpret
from why.explainer import Explainer


def write(exp: Explainer):
    st.title("Feature Importance")
    feature_names = exp.X_test.columns

    imp_types = [
        t.lower()
        for t in st.multiselect(
            "Which feature importance methods would you like to use?",
            options=["Impurity", "Permutation"],
            default="Impurity",
        )
    ]
    if imp_types:
        for i, type in enumerate(imp_types):
            if type == "permutation":
                with st.spinner("Calculating permutation feature importances..."):
                    imp, names = interpret.feature_importance(
                        type,
                        feature_names,
                        estimator=exp.model,
                        X=exp.X_test.values,
                        y=exp.y_test,
                        n_repeats=5,
                        n_jobs=-1,
                    )
                fig = display.plot_permutation_importance(imp, names)
                plt.tight_layout()
                st.pyplot()
            elif type == "impurity":
                imp, names = interpret.feature_importance(
                    type, feature_names, estimator=exp.model
                )
                fig = display.plot_impurity_importance(imp, names)
                plt.tight_layout()
                st.pyplot()
