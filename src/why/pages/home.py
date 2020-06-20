import streamlit as st

from why import display as display
from why.explainer import Explainer
from why.utils import get_data_summary


def write(exp: Explainer) -> None:
    st.markdown("# Why?")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    summary = get_data_summary(exp)
    st.markdown(summary)
    st.dataframe(exp.test.head(100), height=300)

    st.markdown("## Model Performance")
    fig = display.plot_precision_recall_curve(
        exp.y_train, exp.y_test, exp.train_preds, exp.test_preds,
    )

    st.pyplot()
