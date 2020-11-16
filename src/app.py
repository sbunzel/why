import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from why import Explainer, data, models
from why.pages import (
    correlation,
    global_effects,
    home,
    importance,
    local_effects,
    performance,
)

PAGES = {
    "Home": home,
    "Performance": performance,
    "Feature Importance": importance,
    "Feature Correlation": correlation,
    "Global Effects": global_effects,
    "Local Effects": local_effects,
}

st.set_option("deprecation.showfileUploaderEncoding", False)


def main():
    st.sidebar.title("Navigation")
    page = PAGES[st.sidebar.radio("Go to", list(PAGES.keys()))]

    st.sidebar.title("Settings")
    dataset = st.sidebar.selectbox(
        "Select a dataset",
        ["Car Insurance Cold Calls", "Cervical Cancer", "Upload my own data"],
    )

    if dataset == "Upload my own data":
        train_data = st.sidebar.file_uploader(
            "Upload a training dataset in CSV format (comma separated, with headers, UTF-8)",
            type="csv",
            encoding="utf-8",
        )
        if train_data:
            train = pd.read_csv(train_data)
        test_data = st.sidebar.file_uploader(
            "Upload a test dataset in CSV format (comma separated, with headers, UTF-8)",
            type="csv",
            encoding="utf-8",
        )
        if test_data:
            test = pd.read_csv(test_data)
        target = None
    else:
        train, test, target = data.load_data(dataset=dataset)
    if "train" in locals().keys() and "test" in locals().keys():
        if not target:
            target = st.sidebar.selectbox(
                "Select the target column",
                ["No target column selected"] + sorted(test.columns),
            )
        if not target == "No target column selected":
            mode = st.sidebar.selectbox(
                "Select the problem type",
                [
                    "No problem type selected",
                    "Binary Classification",
                    "Multi Class Classification",
                ],
            )
            if not mode == "No problem type selected":
                mode = mode.lower().replace(" ", "_")
                model_type = st.sidebar.selectbox(
                    "Select the model type",
                    ["No model type selected", "Random Forest", "Upload my own model"],
                )
                if not model_type == "No model type selected":
                    if model_type == "Upload my own model":
                        model_buffer = st.sidebar.file_uploader(
                            f"Upload a trained scikit-learn model (saved via joblib.dump() from scikit-learn version {sklearn.__version__}",
                            type=None,
                        )
                        if model_buffer:
                            model = joblib.load(model_buffer)
                    else:
                        model = models.get_model(model_type)
                    if "model" in locals().keys():
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
                        features = (
                            list(set(train.columns) - set(feats_to_remove))
                            if feats_to_remove
                            else None
                        )
                        seed = st.sidebar.number_input(
                            label="Update the random seed",
                            min_value=1,
                            max_value=100,
                            value=42,
                            step=1,
                        )
                        np.random.seed(seed)
                        explainer = Explainer(
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
