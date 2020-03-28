import pandas as pd
import streamlit as st

from .utils import get_root_dir


@st.cache
def get_data(dataset: str):
    if dataset == "Car Insurance Cold Calls":
        datapath = get_root_dir() / "data" / "processed" / "carinsurance"
        return (
            pd.read_csv(datapath / "train.csv"),
            pd.read_csv(datapath / "test.csv"),
            "CarInsurance",
        )
