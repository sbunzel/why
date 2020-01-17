from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

FeatureTable = Union[pd.DataFrame, np.ndarray]
TargetVector = Union[pd.Series, np.ndarray]


def fit_model(model_type: str, X_train: FeatureTable, y_train: TargetVector):
    """Fit a supervised model on X_train and y_train
    
    Args:
        model_type (str): The type of model to fit
        X_train (Union[pd.DataFrame, np.ndarray]): A feature table
        y_train (Union[pd.Series, np.ndarray]): A target vector
    
    Raises:
        NotImplementedError: Raise when an undefined model type is requested
    
    Returns:
        [type]: A fitted model
    """
    if model_type == "Random Forest":
        m = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=3, max_depth=12, n_jobs=-1,
        ).fit(X_train, y_train)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    return m


def get_model_scores(
    m,
    thresh: float,
    X_train: FeatureTable,
    X_valid: FeatureTable,
    y_train: TargetVector,
    y_valid: TargetVector,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate predictions and gather common performance metrics
    
    Args:
        m ([type]): The model to make predictions with
        thresh (float): The classification threshold (classification only)
        X_train (Union[pd.DataFrame, np.ndarray]): Training feature table
        X_valid (Union[pd.DataFrame, np.ndarray]): Validation feature table
        y_train (Union[pd.Series, np.ndarray]): Training target vector
        y_valid (Union[pd.Series, np.ndarray]): Validation target vector
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]: Tuple of metric table, training predictions, validation predictions
    """
    train_pred = m.predict_proba(X_train)[:, 1]
    valid_pred = m.predict_proba(X_valid)[:, 1]
    scores = pd.DataFrame(
        {
            "Accuracy": [
                metrics.accuracy_score(y_train, train_pred > thresh),
                metrics.accuracy_score(y_valid, valid_pred > thresh),
            ],
            "ROC AUC": [
                metrics.roc_auc_score(y_train, train_pred),
                metrics.roc_auc_score(y_valid, valid_pred),
            ],
        },
        index=["Training", "Validation"],
    )
    return scores, train_pred, valid_pred


def get_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, thresh: float
) -> pd.DataFrame:
    """Format a binary classification confusion matrix as a DataFrame
    
    Args:
        y_true (np.ndarray of floats): Array of targets (0 or 1)
        y_pred (np.ndarray of floats): Array of predictions (p1 score between 0 and 1)
        thresh (float): Decision boundary for classifying a prediction as 0 or 1
    
    Returns:
        pd.DataFrame: [description]
    """
    index = ["True 0", "True 1"]
    columns = ["Pred 0", "Pred 1"]
    return pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred > thresh), columns=columns, index=index
    )


def plot_predictions(y_pred: np.ndarray, p_min: float, p_max: float) -> Tuple:
    """Plot model predictions as a histogram and highlight predictions in a selected range
    
    Args:
        y_pred (np.ndarray): Array of predictions
        p_min (float): Lower boundary of predictions to highlight
        p_max (float): Upper boundary of predictions to highlight
    
    Returns:
        Tuple: Matplotlib figure and axes
    """
    fig, ax = plt.subplots()
    ax.hist(y_pred, bins=100, color="grey")
    for i, r in enumerate(ax.patches):
        if p_min <= r.get_x() <= p_max:
            ax.patches[i].set_color("green")
    ax.set_title("Model Prediction")
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_ylabel("Number of predictions")
    plt.tight_layout()
    return fig, ax
