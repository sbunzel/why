import streamlit as st

# import matplotlib.pyplot as plt

from why import display as display
from why import interpret as interpret


def write(state):
    st.title("Feature Importance")
    m, X_valid, y_valid = state["m"], state["X_valid"], state["y_valid"]
    feature_names = X_valid.columns

    imp_types = [t.lower() for t in st.multiselect("Which feature importance methods would you like to use?", options=["Impurity", "Permutation"], default="Impurity")]
    if imp_types:
        # fig, axes = plt.subplots(len(imp_types), 1, figsize=())
        # axes = [axes] if len(imp_types) == 1 else axes
        for i, type in enumerate(imp_types):
            if type == "permutation":
                with st.spinner("Calculating permutation feature importances..."):
                    imp, names = interpret.feature_importance(
                        type, feature_names, estimator=m, X=X_valid.values, y=y_valid, n_repeats=5, n_jobs=-1,
                    )
                # axes[i] = display.plot_permutation_importance(imp, names, axes[i])
                fig, ax = display.plot_permutation_importance(imp, names)
                st.pyplot()
            elif type == "impurity":
                imp, names = interpret.feature_importance(type, feature_names, estimator=m)
                # axes[i] = display.plot_impurity_importance(imp, names, axes[i])
                fig, ax = display.plot_impurity_importance(imp, names)
                st.pyplot()
