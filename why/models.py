import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def fit_model(model_type: str, X_train, y_train):
    if model_type == "Random Forest":
        m = RandomForestClassifier(
            n_estimators=20, min_samples_leaf=3, max_depth=12, n_jobs=-1,
        ).fit(X_train, y_train)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented yet.")
    return m


def get_model_scores(m, thresh, X_train, X_valid, y_train, y_valid):
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
    index = ["True 0", "True 1"]
    columns = ["Pred 0", "Pred 1"]
    return pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred > thresh), columns=columns, index=index
    )
