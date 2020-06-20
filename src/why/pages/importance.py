import streamlit as st

import matplotlib.pyplot as plt

from why import interpret as interpret
from why.explainer import Explainer


def write(exp: Explainer):
    st.title("Feature Importance")

    imp_types = [
        e.replace(" ", "")
        for e in st.multiselect(
            "Which feature importance methods would you like to use?",
            options=["Impurity Importance", "Permutation Importance"],
            default=None,
        )
    ]
    dataset = st.radio(
        label="Choose the dataset to calculate feature importances on",
        options=["train", "test"],
        format_func=lambda x: x.title(),
    )
    if imp_types:
        for imp_type in imp_types:
            with st.spinner(f"Calculating {imp_type}..."):
                importance = getattr(interpret, imp_type)(exp=exp, dataset=dataset)
            fig = importance.plot(top_n=15)
            plt.tight_layout()
            st.pyplot(fig)
