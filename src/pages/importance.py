import streamlit as st

from why import interpret


def write(state):
    st.title("Feature Importance")

    m, X_valid, y_valid = state["m"], state["X_valid"], state["y_valid"]

    with st.spinner("Calculating permutation feature importances..."):
        imp = interpret.feature_importance(
            m, X_valid.values, y_valid, n_repeats=5, n_jobs=-1
        )
    fig, ax = interpret.plot_importance(imp, X_valid.columns)
    st.pyplot()
