import pandas as pd
import streamlit as st

import why.pages.home as home
import why.pages.importance as importance
import why.pages.global_effects as global_effects
import why.pages.local_effects as local_effects
import why.data as data

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
        "Select a dataset", ["Car Insurance Cold Calls", "Upload my own data"]
    )

    if dataset == "Upload my own data":
        test_data = st.sidebar.file_uploader(
            "Upload a test dataset in CSV format (',' separated, with headers, UTF-8)",
            type="csv",
        )
        if test_data:
            test = pd.read_csv(test_data)
        train_data = st.sidebar.file_uploader(
            "Upload a training dataset in CSV format", type="csv",
        )
        if train_data:
            train = pd.read_csv(train_data)
        target = None
    else:
        train, test, target = data.get_data(dataset)
    if "train" in locals().keys() and "test" in locals().keys():
        if not target:
            target = st.sidebar.selectbox(
                "Select the target column",
                ["No target column selected"] + sorted(test.columns),
            )
        if not target == "No target column selected":
            mode = (
                st.sidebar.selectbox(
                    "Select the problem type",
                    ["No problem type selected", "Binary Classification"],
                )
                .lower()
                .replace(" ", "_")
            )
            if not mode == "No problem type selected":
                model_type = st.sidebar.selectbox(
                    "Select the model type",
                    ["No model type selected", "Random Forest"],
                )
                if not model_type == "No model type selected":
                    random_feature = st.sidebar.radio(
                        "Insert a random feature to investigate its effect on the explanations",
                        ["No", "Yes"],
                        key="random_feature",
                    )
                    random_feature = True if random_feature == "Yes" else False
                    settings = {k: v for k, v in locals().items()}
                    session = home.get_state(settings)
                    page.write(session)


if __name__ == "__main__":
    main()
