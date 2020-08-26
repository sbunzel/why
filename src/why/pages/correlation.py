import matplotlib.pyplot as plt
import streamlit as st

from why import Explainer
from why import interpret


def write(exp: Explainer):
    st.title("Feature Correlation")
    st.markdown("#### Correlations between Features")
    corr_methods = ["spearman", "kendall", "pearson"]
    method = st.selectbox(
        "Please select a method to calculate correlations with",
        ["Don't calculate feature correlations"] + corr_methods,
        format_func=lambda x: x.title() if x in corr_methods else x,
    )
    if method in corr_methods:
        feat_corr = interpret.FeatureCorrelation(exp)
        fig = feat_corr.calculate_correlation(
            method=method, sample_size=1000
        ).plot_dendrogram()
        plt.tight_layout()
        st.pyplot(fig)
