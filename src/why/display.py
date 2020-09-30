import re
from typing import Optional
import altair as alt
import pandas as pd


def plot_predictions(df: pd.DataFrame, p_min: Optional[float], p_max: Optional[float]):
    if df["target"].nunique() == 1:
        df = df.assign(focus=lambda df: df["prediction"].between(p_min, p_max))
        color = alt.Color("focus:N", legend=None, scale=alt.Scale(scheme="tableau10"),)
    else:
        color = alt.Color("target:N", scale=alt.Scale(scheme="tableau10"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X(
                "prediction:Q",
                bin=alt.Bin(step=0.01),
                scale=alt.Scale(domain=(0, 1)),
                title="Predicted Probability",
            ),
            y=alt.Y("count()", title="Number of Predictions"),
            color=color,
            tooltip=["target"],
        )
        .properties(
            width="container", height=300, title="Distribution of Model Predictions"
        )
        .configure_title(fontSize=14, offset=10, orient="top", anchor="middle")
    )


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
