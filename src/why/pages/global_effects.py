import matplotlib.pyplot as plt
import streamlit as st

from why import Explainer, interpret


def write(exp: Explainer):
    st.title("Global Effects")
    st.markdown("#### Partial Dependence of Predictions on a Feature")
    feat = st.selectbox(
        "Please select a feature",
        ["Don't plot partial dependence"] + sorted(exp.X_train.columns),
    )
    if not feat == "Don't plot partial dependence":
        pdp_types = [
            e.replace(" ", "")
            for e in st.multiselect(
                "Which partial dependence method would you like to use?",
                options=["Model Dependent PD", "Model Independent PD"],
                default=None,
            )
        ]
        dataset = st.selectbox(
            "Please select dataset on which to calculate partial depedence",
            ["test", "train"],
            format_func=lambda x: x.title(),
        )

        for pdp_type in pdp_types:
            pdp = getattr(interpret, pdp_type)(exp=exp)
            fig = pdp.plot(dataset=dataset, feature=feat)
            plt.tight_layout()
            st.pyplot(fig)
