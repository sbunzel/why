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

    st.sidebar.title("Settings")
    dataset = st.sidebar.selectbox(
        "Please select a dataset", ["Car Insurance Cold Calls"]
    )
    np.random.seed(42)
    random_feature = st.sidebar.radio(
        "Insert a random feature to investigate its effect on the explanations",
        ["No", "Yes"],
        key="random_feature",
    )
    random_feature = True if random_feature == "Yes" else False
    model_type = st.sidebar.selectbox(
        "Change the model type to understand its impact on the explanations",
        ["Random Forest"],
    )
    settings = {k: v for k, v in locals().items()}
    session = home.get_state(settings)
    page.write(session)


if __name__ == "__main__":
    main()
