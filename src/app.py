import pandas as pd
import streamlit as st

import why.pages.home as home
import why.pages.importance as importance
import why.pages.global_effects as global_effects
import why.pages.local_effects as local_effects
import why.data as data
import why.models as models
import why.explainer as exp

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
        train_data = st.sidebar.file_uploader(
            "Upload a training dataset in CSV format (comma separated, with headers, UTF-8)",
            type="csv",
        )
        if train_data:
            train = pd.read_csv(train_data)
        test_data = st.sidebar.file_uploader(
            "Upload a test dataset in CSV format (comma separated, with headers, UTF-8)",
            type="csv",
        )
        if test_data:
            test = pd.read_csv(test_data)
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
            mode = st.sidebar.selectbox(
                "Select the problem type",
                ["No problem type selected", "Binary Classification"],
            )
            if not mode == "No problem type selected":
                mode = mode.lower().replace(" ", "_")
                model_type = st.sidebar.selectbox(
                    "Select the model type",
                    ["No model type selected", "Random Forest", "Upload my own model"],
                )
                if not model_type == "No model type selected":
                    model = models.get_model(model_type)
                    random_feature = st.sidebar.radio(
                        "Insert a random feature to investigate its effect on the explanations",
                        ["No", "Yes"],
                        key="random_feature",
                    )
                    random_feature = True if random_feature == "Yes" else False
                    feats_to_remove = st.sidebar.multiselect(
                        "Select features to remove from the model",
                        options=sorted(train.columns),
                        default=None,
                    )
                    features = list(set(train.columns) - set(feats_to_remove))
                    explainer = exp.Explainer(
                        train=train,
                        test=test,
                        target=target,
                        model=model,
                        features=features,
                        mode=mode,
                        random_feature=random_feature,
                    )
                    page.write(explainer)


if __name__ == "__main__":
    main()
