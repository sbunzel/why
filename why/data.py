from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

__all__ = ["get_data", "CarInsurance"]


@st.cache
def get_data(
    dataset: str, config: Dict[str, Any], datapath: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "car_insurance_cold_calls":
        data = CarInsurance(config, datapath)
    else:
        raise NotImplementedError(
            f"No data set for {dataset.replace('_', ' ').title()} available."
        )
    return data.train, data.test


class CarInsurance:
    def __init__(self, config: Dict[str, Any], datapath: Path) -> None:
        """Initialize the car insurance data set
        
        Args:
            config (Dict[str, Any]): Configuration file for the car insurance data
            datapath (Path): Path to the folder containing the car insurance data
        """
        self.config = config
        self.datapath = datapath
        self.train, self.test = self.prepare_data()

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read the insurance data and apply the data transformation pipeline to train and test
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The prepared car insurance train and test set
        """
        train, test = self.get_data()
        return train.pipe(self.preparation_pipe), test.pipe(self.preparation_pipe)

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read the car insurance data into a Pandas DataFrame
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The raw car insurance train and test set
        """
        date_parser = partial(pd.to_datetime, format="%H:%M:%S")
        train = pd.read_csv(
            self.datapath / "carInsurance_train.csv",
            parse_dates=["CallStart", "CallEnd"],
            date_parser=date_parser,
        ).set_index("Id")
        test = pd.read_csv(
            self.datapath / "carInsurance_test.csv",
            parse_dates=["CallStart", "CallEnd"],
            date_parser=date_parser,
        ).set_index("Id")
        return train, test

    def preparation_pipe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine preprocessing steps for the car insurance data in one function
        
        Args:
            df (pd.DataFrame): Raw car insurance data
        
        Returns:
            pd.DataFrame: Transformed car insurance data
        """
        return (
            df.pipe(self.add_call_time_features)
            .pipe(self.map_binary_cols)
            .rename(self.config["colmapper"], axis=1)
            .drop(["CallStart", "CallEnd"], axis=1)
        )

    def add_call_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add new features based on CallStart and CallEnd
        
        Args:
            df (pd.DataFrame): Car insurance data
        
        Returns:
            pd.DataFrame: Enhanced car insurance data
        """
        return df.assign(
            LastCallDurationSecs=lambda df: (
                df["CallEnd"] - df["CallStart"]
            ).dt.seconds,
            LastCallHourOfDay=lambda df: df["CallStart"].dt.hour,
        )

    def map_binary_cols(
        self, df: pd.DataFrame, cols: List[str] = ["CarLoan", "Default", "HHInsurance"]
    ) -> pd.DataFrame:
        """Map binary columns to more expressive values
        
        Args:
            df (pd.DataFrame): Car insurance data
            cols (List[str], optional): Column names of binary columns to transform.
                Defaults to ["CarLoan", "Default", "HHInsurance"].
        
        Returns:
            pd.DataFrame: Transformed car insurance data
        """
        df = df.copy()
        for c in cols:
            df[c] = df[c].map({0: "no", 1: "yes"})
        return df
