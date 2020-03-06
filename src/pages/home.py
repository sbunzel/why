from pathlib import Path
from typing import Any, Dict

import numpy as np
import streamlit as st

from why import data as data
from why import display as display
from why import utils as utils
from why import models as models


def get_state(settings: Dict[str, Any]) -> Dict[Dict[str, Any], Dict[str, Any]]:
    config = utils.get_config(settings["dataset"])
    X_train, X_valid, y_train, y_valid, summary = data.get_data(
        dataset=settings["dataset"],
        config=config,
        datapath=(Path(__file__).parent.parent.parent / "data" / "raw").resolve(),
    )

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
    st.markdown(session["summary"])
    st.dataframe(session["X_train"], height=300)

    st.markdown("## Model Performance")
    fig = display.plot_precision_recall_curve(
        session["y_train"],
        session["y_valid"],
        session["train_pred"],
        session["valid_pred"],
    )

    st.pyplot()
