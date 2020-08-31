import matplotlib.pyplot as plt
import streamlit as st

from why import Explainer, interpret


def write(exp: Explainer):
    st.title("Global Effects")
    st.markdown("#### Partial Dependence of Predictions on a Feature")
    if exp.mode == "multi_class_classification":
        pdp_target = st.selectbox(
            "Please select the target class for which to compute PDP",
            sorted(exp.model.classes_),
        )
    else:
        pdp_target = None
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
            if pdp_type == "ModelDependentPD" or exp.mode == "binary_classification":
                pdp = getattr(interpret, pdp_type)(exp=exp)
                fig = pdp.plot(dataset=dataset, feature=feat, target=pdp_target)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning(
                    f"'{pdp_type}' has not been implemented for mode '{exp.mode}' yet."
                )
