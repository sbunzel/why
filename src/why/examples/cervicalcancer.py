from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def load_cervical_cancer() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect and prepare the cervical cancer risk factors data set from the UCI Machine Learning Repository.
    A description of the dataset is available here: https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and test data.
    """
    COLS = [
        "Age",
        "Number of sexual partners",
        "First sexual intercourse",
        "Num of pregnancies",
        "Smokes",
        "Smokes (years)",
        "Smokes (packs/year)",
        "Hormonal Contraceptives",
        "Hormonal Contraceptives (years)",
        "IUD",
        "IUD (years)",
        "STDs",
        "STDs (number)",
        "STDs: Number of diagnosis",
        "STDs: Time since first diagnosis",
        "STDs: Time since last diagnosis",
        "Biopsy",
    ]
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv",
        usecols=COLS,
        na_values="?",
    )
    X, y = df.drop(columns="Biopsy"), df["Biopsy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    imputer = SimpleImputer(strategy="most_frequent", add_indicator=True).fit(X_train)
    feature_names = list(X_train.columns) + [
        f"MISSING: {c}" for c in X_train.columns[imputer.indicator_.features_]
    ]
    train = pd.DataFrame(imputer.transform(X_train), columns=feature_names).assign(
        Biopsy=y_train.values
    )
    test = pd.DataFrame(imputer.transform(X_test), columns=feature_names).assign(
        Biopsy=y_test.values
    )
    return train, test
