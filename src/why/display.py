import re

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_predictions(y_pred: np.ndarray, p_min: float, p_max: float) -> Figure:
    """Plots model predictions as a histogram and highlight predictions in a selected range.
    
    Args:
        y_pred (np.ndarray): Array of predictions.
        p_min (float): Lower boundary of predictions to highlight.
        p_max (float): Upper boundary of predictions to highlight.
    
    Returns:
        Figure: Matplotlib figure of prediction histogram.
    """
    fig, ax = plt.subplots()
    ax.hist(y_pred, bins=100, color="grey")
    for i, r in enumerate(ax.patches):
        if p_min <= r.get_x() <= p_max:
            ax.patches[i].set_color("#90ee90")
    ax.set_title("Model Prediction")
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Number of predictions")
    plt.tight_layout()
    return fig


def style_local_explanations(
    feat_values: pd.DataFrame, min_pred: float = 0.0, max_pred: float = 1.0
) -> pd.DataFrame:
    """Applies Pandas Stylers to the local feature effects to distinguish positive and negative effects.

    Args:
        feat_values (pd.DataFrame): DataFrame of local feature effects and predictions.

    Returns:
        pd.DataFrame: Styled DataFrame.
    """
    return feat_values.style.applymap(
        _color_by_sign, subset=list(set(feat_values.columns) - set(["Prediction"]))
    ).background_gradient(
        cmap="RdYlBu", axis="index", subset="Prediction", vmin=min_pred, vmax=max_pred
    )


def _color_by_sign(effect_string: str) -> str:
    """Colors individual feature effect strings based on the direction of the effect (i.e., negative or positive).
    Each input corresponds to an individual cell in the local effects DataFrame.

    Args:
        effect_string (str): A string describing the effect of the form %feature = %feature_value ( %effect_size ).

    Returns:
        str: A CSS styling of the cell.
    """
    try:
        # Extract the sign from the effect size
        sign = re.search("(\( ([\+\-])\d\.)", effect_string).group(2)
    except Exception:
        sign = None
    if sign == "-":
        color = "#ee9090"
    elif sign == "+":
        color = "#90ee90"
    else:
        color = "#a9a9a9"
    return f"border-left: 8px solid {color}"
