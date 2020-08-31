from matplotlib.figure import Figure
import streamlit as st

from why import Explainer
from why.evaluate import BinaryClassEvaluator, MultiClassEvaluator

EVALUATORS = {
    "binary_classification": BinaryClassEvaluator,
    "multi_class_classification": MultiClassEvaluator,
}


def write(exp: Explainer) -> None:
    st.markdown("# Why?")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    st.markdown(exp.get_data_summary())
    st.dataframe(exp.test.head(100), height=300)

    st.markdown("## Model Performance")
    evaluator = EVALUATORS[exp.mode](exp)
    method = st.selectbox(
        "Select evaluation method",
        evaluator.methods,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    evaluation = getattr(evaluator, method)()
    if isinstance(evaluation, Figure):
        st.pyplot(evaluation)
    else:
        st.write(evaluation)
