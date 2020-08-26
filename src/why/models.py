from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


FeatureTable = Union[pd.DataFrame, np.ndarray]
TargetVector = Union[pd.Series, np.ndarray]


def get_model(model_type: str) -> BaseEstimator:
    """Get a model instance based on the model type selected.
    
    Args:
        model_type (str): The type of model to instantiate.
    
    Raises:
        NotImplementedError: Raise when an undefined model type is requested.
    
    Returns:
        BaseEstimator: A model instance.
    """
    if model_type == "Random Forest":
        m = RandomForestClassifier(
            n_estimators=20,
            min_samples_leaf=3,
            max_depth=12,
            n_jobs=-1,
            # random_state=42,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")
    return m
