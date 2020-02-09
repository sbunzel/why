import numpy as np
import streamlit as st

# from pages import home, importance, global_effects, local_effects
import pages.home as home
import pages.importance as importance
import pages.global_effects as global_effects
import pages.local_effects as local_effects

PAGES = {
    "Home": home,
    "Feature Importance": importance,
    "Global Effects": global_effects,
    "Local Effects": local_effects,
}


def main():
    st.sidebar.title("Navigation")
    page = PAGES[st.sidebar.radio("Go to", list(PAGES.keys()))]

    st.sidebar.title("Settings and Details")
    st.sidebar.markdown("## General")
    dataset = st.sidebar.selectbox(
        "Please select a dataset", ["Car Insurance Cold Calls"]
    )
    remove_leaky = st.sidebar.radio(
        "Exclude features derived from the last contact - they might be leaky!",
        ["No", "Yes"],
    )
    remove_leaky = True if remove_leaky == "Yes" else False
    min_corr = st.sidebar.slider(
        "Mininum correlation to drop correlated features", 0.5, 1.0, 1.0, step=0.05
    )
    seed = st.sidebar.number_input(
        "Change the seed to investigate how random effects might impact the model and explanations",
        value=42,
    )
    np.random.seed(seed)
    st.sidebar.markdown("## Model Type")
    model_type = st.sidebar.selectbox(
        "Change the model type to understand its impact on the explanations",
        ["Random Forest"],
    )
    st.sidebar.markdown("## Model Performance")
    threshold = st.sidebar.slider(
        "The classification threshold used to calculate accuracy and derive the confusion matrix",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    session = {k: v for k, v in locals().items()}
    state = home.get_state(session)
    page.write(state)


if __name__ == "__main__":
    main()
