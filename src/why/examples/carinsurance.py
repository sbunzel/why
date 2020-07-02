from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

COLMAPPER = {
    "Marital": "MaritalStatus",
    "Education": "EducationLevel",
    "Default": "HasCreditDefault",
    "HHInsurance": "HasHHInsurance",
    "CarLoan": "HasCarLoan",
    "Communication": "CommunicationType",
    "LastContactDay": "LastContactDayOfMonth",
    "NoOfContacts": "NumContactsCurrentCampaign",
    "PrevAttempts": "NumAttemptsPrevCampaign",
    "Outcome": "PrevCampaignOutcome",
}
CATCOLS = [
    "Job",
    "MaritalStatus",
    "EducationLevel",
    "HasCreditDefault",
    "HasHHInsurance",
    "HasCarLoan",
    "CommunicationType",
    "LastContactMonth",
    "PrevCampaignOutcome",
]
NUMCOLS = [
    "Age",
    "Balance",
    "LastContactDayOfMonth",
    "NumContactsCurrentCampaign",
    "DaysPassed",
    "NumAttemptsPrevCampaign",
    "LastCallDurationSecs",
    "LastCallHourOfDay",
]
TARGET = "CarInsurance"


def load_car_insurance(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the car insurance cold calls dataset.
    A description of the dataset is available here: https://www.kaggle.com/kondla/carinsurance

    Args:
        file_path (Path): Path to a local copy of the Kaggle training set `carInsurance_train.csv' 

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and test data.
    """
    df = (
        pd.read_csv(
            file_path,
            parse_dates=["CallStart", "CallEnd"],
            date_parser=partial(pd.to_datetime, format="%H:%M:%S"),
        )
        .set_index("Id")
        .pipe(_add_call_time_features)
        .pipe(_map_binary_cols)
        .rename(COLMAPPER, axis=1)
        .drop(["CallStart", "CallEnd"], axis=1)
    )
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    transformer = ColumnTransformer(
        [("cat", cat_pipe, CATCOLS), ("num", num_pipe, NUMCOLS)], sparse_threshold=0,
    ).fit(X_train)
    cat_names, feature_names = _get_feature_names(transformer)
    train = (
        pd.DataFrame(
            transformer.transform(X_train), columns=feature_names, index=y_train.index
        )
        .astype({feat: "int" for feat in cat_names})
        .assign(**{TARGET: y_train})
    )
    test = (
        pd.DataFrame(
            transformer.transform(X_test), columns=feature_names, index=y_test.index
        )
        .astype({feat: "int" for feat in cat_names})
        .assign(**{TARGET: y_test})
    )
    return train, test


def _add_call_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features based on CallStart and CallEnd

    Args:
        df (pd.DataFrame): Car insurance data

    Returns:
        pd.DataFrame: Enhanced car insurance data
    """
    return df.assign(
        LastCallDurationSecs=lambda df: (df["CallEnd"] - df["CallStart"]).dt.seconds,
        LastCallHourOfDay=lambda df: df["CallStart"].dt.hour,
    )


def _map_binary_cols(
    df: pd.DataFrame, cols: List[str] = ["CarLoan", "Default", "HHInsurance"]
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


def _get_feature_names(
    transformer: ColumnTransformer,
) -> Tuple[Sequence[str], Sequence[str]]:
    """Extract the feature names of the transformed data from a fitted ColumnTransformer

    Returns:
        Tuple[Sequence[str], Sequence[str]]: Categorical feature names, all feature names
    """
    ohe = transformer.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names(input_features=CATCOLS)
    feature_names = np.r_[cat_feature_names, NUMCOLS]
    return cat_feature_names, feature_names
