import re

from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_predictions(y_pred: np.ndarray, p_min: float, p_max: float) -> Tuple:
    """Plot model predictions as a histogram and highlight predictions in a selected range
    
    Args:
        y_pred (np.ndarray): Array of predictions
        p_min (float): Lower boundary of predictions to highlight
        p_max (float): Upper boundary of predictions to highlight
    
    Returns:
        Tuple: Matplotlib figure and axes
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
    return fig, ax


def plot_precision_recall_curve(y_train, y_valid, train_pred, valid_pred):
    train_pr, train_rc, _ = precision_recall_curve(y_train, train_pred)
    train_ap = average_precision_score(y_train, train_pred)
    valid_pr, valid_rc, _ = precision_recall_curve(y_valid, valid_pred)
    valid_ap = average_precision_score(y_valid, valid_pred)

    fig, ax = plt.subplots()
    ax.plot(train_rc, train_pr, label=f"Train Average Precision = {train_ap:.2f}")
    ax.plot(valid_rc, valid_pr, label=f"Test Average Precision = {valid_ap:.2f}")
    ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    plt.legend()
    return fig


def color_by_sign(val):
    try:
        sign = re.search("(\( ([\+\-])\d\.)", val).group(2)
    except Exception:
        sign = None
    if sign == "-":
        color = "#ee9090"
    elif sign == "+":
        color = "#90ee90"
    else:
        color = "#a9a9a9"
    return f"border-left: 8px solid {color}"


def format_local_explanations(feat_values: pd.DataFrame) -> pd.DataFrame.style:
    return feat_values.style.applymap(
        color_by_sign, subset=list(set(feat_values.columns) - set(["Prediction"]))
    ).background_gradient(
        cmap="RdYlGn", axis="index", subset="Prediction"
    )  # TODO: This should be red for small p1s
