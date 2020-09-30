import streamlit as st

from why import Explainer


def write(exp: Explainer) -> None:
    st.markdown("# Why?")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    st.markdown(exp.get_data_summary())
    st.dataframe(exp.test.head(100), height=300)
