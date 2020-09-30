import altair as alt
from matplotlib.figure import Figure
import pandas as pd
import streamlit as st

from why import Explainer
from why.evaluate import evaluators
from why import display


def write(exp: Explainer) -> None:
    st.title("Model Performance")
    distribution_plot = st.empty()
    if exp.mode == "binary_classification":
        target = None
        threshold = st.slider(
            "Set the threshold for classifying an observation as class 1",
            0.0,
            1.0,
            0.5,
        )
    else:
        target = st.selectbox(
            "Select the target class to inspect",
            ["Show all"] + list(exp.y_test.unique()),
        )
        threshold = None
    if not target == "Show all":
        preds = exp.get_class_test_preds(class_name=target)
        pred_df = pd.DataFrame(data={"target": 1, "prediction": preds})
    else:
        pred_df = pd.DataFrame(exp.test_preds, columns=exp.y_train.unique()).melt(
            var_name="target", value_name="prediction"
        )
    chart = display.plot_predictions(pred_df, p_min=threshold, p_max=1)
    distribution_plot.altair_chart(chart, use_container_width=True)

    evaluator = evaluators[exp.mode](exp)
    method = st.selectbox(
        "Select evaluation method",
        evaluator.methods,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    evaluation = getattr(evaluator, method)(target=target, threshold=threshold)
    if isinstance(evaluation, Figure):
        st.pyplot(evaluation)
    elif isinstance(evaluation, alt.Chart):
        st.altair_chart(evaluation, use_container_width=True)
    else:
        st.write(evaluation)
