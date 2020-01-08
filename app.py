import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.inspection import permutation_importance, plot_partial_dependence
from stratx.partdep import plot_stratpd

from why import data, preprocessing


st.title("Why?")
st.header("An exploration into the world of interpretable machine learning")

st.subheader("The Dataset")
# Let the user define the dataset to be used on collect it based on their input
dataset = (
    st.selectbox("Please select a dataset", ["Car Insurance Cold Calls"])
    .lower()
    .replace(" ", "_")
)

config_path = (Path(__file__).parent / "resources" / f"{dataset}.json").resolve()
with open(config_path, mode="r") as f:
    config = json.load(f)
train, test = data.get_data(
    dataset, config, (Path(__file__).parent / "data" / "raw").resolve()
)
n_rows = train.shape[0] + test.shape[0]
n_cols = train.shape[1] + test.shape[1]
st.markdown(
    f"There are **{n_rows}** observations and **{n_cols - 1}** features in this dataset. The target variable is **{config['target']}.**"
)

st.dataframe(train, height=150)

# Prepare the data for modeling
trans = preprocessing.InsuranceTransformer(config)
X_train, X_valid, y_train, y_valid = trans.prepare_train_valid(train)

# Limit the dataset
st.sidebar.subheader("Filters")
original_features = sorted(set(train.columns) - set(config["target"]))
drop_cols = st.sidebar.multiselect(
    "Please select features to exclude", options=original_features
)
to_drop = [c for c in X_train.columns if any(d in c for d in drop_cols)]
X_train = X_train.drop(columns=to_drop, errors="ignore")
X_valid = X_valid.drop(columns=to_drop, errors="ignore")

# Specify and fit a model
st.sidebar.header("Settings and Model Details")
st.sidebar.subheader("Model Type")
model_type = st.sidebar.selectbox("Please select a model type", ["Random Forest"])

with st.spinner(f"Fitting the {model_type} model..."):
    m = RandomForestClassifier(
        n_estimators=20, min_samples_leaf=3, max_depth=12, n_jobs=-1,
    ).fit(X_train, y_train)

# Report model performance
st.sidebar.subheader("Model Performance")
thresh = st.sidebar.slider(
    "Please select a classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

train_pred = m.predict_proba(X_train)[:, 1]
valid_pred = m.predict_proba(X_valid)[:, 1]
scores = pd.DataFrame(
    {
        "Accuracy": [
            metrics.accuracy_score(y_train, train_pred > thresh),
            metrics.accuracy_score(y_valid, valid_pred > thresh),
        ],
        "ROC AUC": [
            metrics.roc_auc_score(y_train, train_pred),
            metrics.roc_auc_score(y_valid, valid_pred),
        ],
    },
    index=["Training", "Validation"],
)

st.sidebar.markdown("**Metrics**")
st.sidebar.dataframe(scores)

st.sidebar.markdown("**Confusion Matrix**")
index = ["True 0", "True 1"]
columns = ["Pred 0", "Pred 1"]
conf_matrix = pd.DataFrame(
    metrics.confusion_matrix(y_valid, valid_pred > thresh), columns=columns, index=index
)
st.sidebar.dataframe(conf_matrix)

# Show interpretation results
st.subheader("Understanding the Model")
feature_names = X_train.columns

# Feature importance
st.markdown("**Most Important Features**")
if st.checkbox("Calculate feature importances"):
    with st.spinner("Calculating the feature importance..."):
        permutation_feat_imp_valid = permutation_importance(
            m, X_valid.values, y_valid, n_repeats=10, n_jobs=-1
        )

    sorted_idx = permutation_feat_imp_valid.importances_mean.argsort()[-15:]
    fig, ax = plt.subplots()
    ax.boxplot(
        permutation_feat_imp_valid.importances[sorted_idx].T,
        vert=False,
        labels=feature_names[sorted_idx],
    )
    ax.set_title("Permutation Importances (on the validation set)")
    plt.tight_layout()
    st.pyplot()

# Partial Dependence Plots
st.markdown("**Partial Dependence of Predictions on a Feature**")
feat = st.selectbox("Please select a feature", sorted(X_train.columns))
if st.checkbox("Calculate model-based partial dependence"):
    plot_partial_dependence(
        m, X_valid, features=[feat], feature_names=feature_names, grid_resolution=20,
    )
    plt.tight_layout()
    st.pyplot()

if st.checkbox("Calculate model-free partial dependence"):
    plot_stratpd(
        X_valid, y_valid, feat, config["target"], min_samples_leaf=20, ntrees=10,
    )
    plt.tight_layout()
    st.pyplot()
