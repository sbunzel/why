from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st

from why import data as data
from why import display as display
from why import preprocessing as preprocessing
from why import utils as utils
from why import models as models


def get_state(settings: Dict[str, Any]) -> Dict[Dict[str, Any], Dict[str, Any]]:
    config = utils.get_config(settings["dataset"])
    train, test = data.get_data(
        settings["dataset"],
        config,
        (Path(__file__).parent.parent.parent / "data" / "raw").resolve(),
    )

    if settings["remove_leaky"]:
        to_drop = config["last_contact_features"]
        config["catcols"] = list(set(config["catcols"]) - set(to_drop))
        config["numcols"] = list(set(config["numcols"]) - set(to_drop))
        train = train.drop(columns=to_drop)
        test = test.drop(columns=to_drop)

    trans = preprocessing.InsuranceTransformer(config)
    X_train, X_valid, y_train, y_valid = trans.prepare_train_valid(train)

    min_corr = settings["min_corr"]
    X_train = X_train.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)
    X_valid = X_valid.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)

    if settings["random_feature"]:
        X_train["RANDOM_NUM"] = np.random.rand(X_train.shape[0])
        X_valid["RANDOM_NUM"] = np.random.rand(X_valid.shape[0])

    model_type = settings["model_type"]
    with st.spinner(f"Fitting the {model_type} model..."):
        m = models.fit_model(model_type, X_train, y_train)

    train_pred, valid_pred = models.get_model_scores(
        m, X_train, X_valid, y_train, y_valid
    )
    session = {k: v for k, v in locals().items()}
    return {**settings, **session}


def write(session: Dict[Dict[str, Any], Dict[str, Any]]) -> None:
    st.markdown("# Why?")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    st.markdown(utils.get_data_summary(session["train"], session["test"], session["config"]["target"]))
    st.dataframe(session["train"], height=300)

    st.markdown("## Model Performance")
    fig = display.plot_precision_recall_curve(session["y_train"], session["y_valid"], session["train_pred"], session["valid_pred"])
    
    st.pyplot()
