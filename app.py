from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.inspection import plot_partial_dependence
from stratx.partdep import plot_stratpd

# from why import data, preprocessing, utils
import why.data as data
import why.preprocessing as preprocessing
import why.utils as utils
import why.models as models
import why.interpret as interpret
import why.display as display

st.markdown("# Why?")
st.markdown("**_An exploration into the world of interpretable machine learning_**")

# Read the configuration and the dataset selected by the user
st.markdown("## The Dataset")
dataset = st.selectbox("Please select a dataset", ["Car Insurance Cold Calls"])
config = utils.get_config(dataset)
train, test = data.get_data(
    dataset, config, (Path(__file__).parent / "data" / "raw").resolve()
)
st.markdown(utils.get_data_summary(train, test, config["target"]))
st.dataframe(train, height=150)

# Prepare the data for modeling
trans = preprocessing.InsuranceTransformer(config)
X_train, X_valid, y_train, y_valid = trans.prepare_train_valid(train)

# Filter the dataset
st.sidebar.markdown("# Filters")
to_drop = st.sidebar.multiselect(
    "Please select features to exclude",
    options=sorted(set(train.columns) - set(config["target"])),
)
to_drop = [c for c in X_train.columns if any(d in c for d in to_drop)]
X_train = X_train.drop(columns=to_drop, errors="ignore")
X_valid = X_valid.drop(columns=to_drop, errors="ignore")

# Specify and fit a model
st.sidebar.markdown("# Settings and Model Details")
seed = st.sidebar.number_input("Please choose a seed", value=42)
np.random.seed(seed)
st.sidebar.markdown("## Model Type")
model_type = st.sidebar.selectbox("Please select a model type", ["Random Forest"])
with st.spinner(f"Fitting the {model_type} model..."):
    m = models.fit_model(model_type, X_train, y_train)

# Report model performance
st.sidebar.markdown("## Model Performance")
thresh = st.sidebar.slider(
    "Please select a classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)
scores, train_pred, valid_pred = models.get_model_scores(
    m, thresh, X_train, X_valid, y_train, y_valid
)
st.sidebar.markdown("### Metrics")
st.sidebar.dataframe(scores)
st.sidebar.markdown("### Confusion Matrix")
conf_matrix = models.get_confusion_matrix(y_valid, valid_pred, thresh)
st.sidebar.dataframe(conf_matrix)

# Showcase and compare interpretation methods
st.markdown("## Understanding the Model")
feature_names = X_train.columns

st.markdown("### Global Effects")

# Feature importance
st.markdown("#### Most Important Features")
if st.checkbox("Calculate feature importances"):
    with st.spinner("Calculating the feature importance..."):
        imp = interpret.feature_importance(
            m, X_valid.values, y_valid, n_repeats=10, n_jobs=-1
        )
    fig, ax = interpret.plot_importance(imp, feature_names)
    st.pyplot()

# Partial Dependence Plots
st.markdown("#### Partial Dependence of Predictions on a Feature")
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

st.markdown("### Local Effects")

st.markdown("#### Distribution of Model Predictions")
distribution_plot = st.empty()
p_min, p_max = st.slider(
    "Select range of predictions to explain",
    min_value=0.0,
    max_value=1.0,
    value=(0.9, 1.0),
    step=0.01,
)
fig, ax = display.plot_predictions(valid_pred, p_min, p_max)
distribution_plot.pyplot()

selected_ids = np.where(np.logical_and(valid_pred >= p_min, valid_pred <= p_max))[0]
n_feats = st.number_input(
    "Please select the number of features to show", value=3, min_value=3, max_value=10
)

st.markdown(f"#### Shapley Values for Top {n_feats} Features")
shap_values = interpret.shap_values(m, X_valid)
feat_values, local_effects = interpret.shap_feature_values(
    m, X_valid, shap_values, selected_ids, n_feats=n_feats
)
st.table(display.format_local_explanations(feat_values))
