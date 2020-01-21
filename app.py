from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.inspection import plot_partial_dependence

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
st.markdown("## Inspecting the Data")
dataset = st.selectbox("Please select a dataset", ["Car Insurance Cold Calls"])
config = utils.get_config(dataset)
train, test = data.get_data(
    dataset, config, (Path(__file__).parent / "data" / "raw").resolve()
)

# Summarize and show the data, potentially removing leaky features
data_description = st.empty()
sample_data = st.empty()
if st.checkbox("Exclude features derived from the last contact - they might be leaky!"):
    to_drop = config["last_contact_features"]
    config["catcols"] = list(set(config["catcols"]) - set(to_drop))
    config["numcols"] = list(set(config["numcols"]) - set(to_drop))
    train = train.drop(columns=to_drop)
    test = test.drop(columns=to_drop)
    st.info(f"You excluded the following features: {', '.join(to_drop)}")

data_description.markdown(utils.get_data_summary(train, test, config["target"]))
sample_data.dataframe(train, height=150)

# Filter the dataset
st.sidebar.markdown("# Settings and Details")
trans = preprocessing.InsuranceTransformer(config)
X_train, X_valid, y_train, y_valid = trans.prepare_train_valid(train)

min_corr = st.sidebar.slider(
    "Mininum correlation to drop correlated features", 0.5, 1.0, 1.0, step=0.05
)
X_train = X_train.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)
X_valid = X_valid.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)

# Specify and fit a model
seed = st.sidebar.number_input(
    "Change the seed to investigate how randomn effects might impact the model and explanations",
    value=42,
)
np.random.seed(seed)
st.sidebar.markdown("## Model Type")
model_type = st.sidebar.selectbox(
    "Change the model type to understand its impact on the explanations",
    ["Random Forest"],
)
with st.spinner(f"Fitting the {model_type} model..."):
    m = models.fit_model(model_type, X_train, y_train)

# Report model performance
st.sidebar.markdown("## Model Performance")
thresh = st.sidebar.slider(
    "The classification threshold used to calculate accuracy and derive the confusion matrix",
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
imp_method = st.selectbox(
    "Please select a feature importance method",
    ["Don't plot feature importances", "Permutation Importance"],
)
if not imp_method == "Don't plot feature importances":
    with st.spinner("Calculating permutation feature importances..."):
        imp = interpret.feature_importance(
            m, X_valid.values, y_valid, n_repeats=10, n_jobs=-1
        )
    fig, ax = interpret.plot_importance(imp, feature_names)
    st.pyplot()

# Partial Dependence Plots
st.markdown("#### Partial Dependence of Predictions on a Feature")
feat = st.selectbox(
    "Please select a feature",
    ["Don't plot partial dependence"] + sorted(X_train.columns),
)
if not feat == "Don't plot partial dependence":
    plot_partial_dependence(
        m, X_valid, features=[feat], feature_names=feature_names, grid_resolution=20
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

local_effects_method = st.selectbox(
    "Please select a local effects method",
    ["Don't display local effects", "Shapley values"],
)
if not local_effects_method == "Don't display local effects":
    n_feats = st.number_input(
        "Please select the number of features to show",
        value=3,
        min_value=3,
        max_value=10,
    )
    st.markdown(f"#### Shapley Values for the Top {n_feats} Features")
    shap_values = interpret.shap_values(m, X_valid)
    feat_values, local_effects = interpret.shap_feature_values(
        m, X_valid, shap_values, selected_ids, n_feats=n_feats
    )
    st.table(display.format_local_explanations(feat_values))
