import matplotlib.pyplot as plt
from sklearn import inspection
import streamlit as st


@st.cache
def feature_importance(*args, **kwargs):
    return inspection.permutation_importance(*args, **kwargs)


def plot_importance(imp, feature_names):
    sorted_idx = imp.importances_mean.argsort()[-15:]
    fig, ax = plt.subplots()
    ax.boxplot(
        imp.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx],
    )
    ax.set_title("Permutation Importances (on the validation set)")
    plt.tight_layout()
    return fig, ax
