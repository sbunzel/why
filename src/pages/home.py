from pathlib import Path

import streamlit as st

from why import data as data
from why import preprocessing as preprocessing
from why import utils as utils
from why import models as models


def get_state(session):
    config = utils.get_config(session["dataset"])
    train, test = data.get_data(
        session["dataset"],
        config,
        (Path(__file__).parent.parent.parent / "data" / "raw").resolve(),
    )

    if session["remove_leaky"]:
        to_drop = config["last_contact_features"]
        config["catcols"] = list(set(config["catcols"]) - set(to_drop))
        config["numcols"] = list(set(config["numcols"]) - set(to_drop))
        train = train.drop(columns=to_drop)
        test = test.drop(columns=to_drop)

    trans = preprocessing.InsuranceTransformer(config)
    X_train, X_valid, y_train, y_valid = trans.prepare_train_valid(train)

    min_corr = session["min_corr"]
    X_train = X_train.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)
    X_valid = X_valid.pipe(preprocessing.remove_high_corrs, min_corr=min_corr)

    model_type = session["model_type"]
    with st.spinner(f"Fitting the {model_type} model..."):
        m = models.fit_model(model_type, X_train, y_train)

    scores, train_pred, valid_pred = models.get_model_scores(
        m, session["threshold"], X_train, X_valid, y_train, y_valid
    )
    state = {k: v for k, v in locals().items()}
    return {**session, **state}


def write(session):
    st.markdown("# Why?")
    st.markdown("**_An exploration into the world of interpretable machine learning_**")

    st.markdown("## The Dataset")
    st.markdown(utils.get_data_summary(session["train"], session["test"], session["config"]["target"]))
    st.dataframe(session["train"], height=300)

    st.sidebar.markdown("### Metrics")
    st.sidebar.dataframe(session["scores"])
    st.sidebar.markdown("### Confusion Matrix")
    conf_matrix = models.get_confusion_matrix(session["y_valid"], session["valid_pred"], session["threshold"])
    st.sidebar.dataframe(conf_matrix)
