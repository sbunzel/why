import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    plot_confusion_matrix,
)

from .explainer import Explainer

__all__ = ["BinaryClassEvaluator"]


class BinaryClassEvaluator:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.methods = ["plot_precision_recall_curve"]

    def plot_precision_recall_curve(self) -> Figure:
        """Calculates and plots the precision recall curve for as set of training and test set predictions.
        
        Returns:
            Figure: Matplotlib figure of train and test precision recall curves.
        """
        train_pr, train_rc, _ = precision_recall_curve(
            self.exp.y_train, self.exp.train_preds
        )
        train_ap = average_precision_score(self.exp.y_train, self.exp.train_preds)
        test_pr, test_rc, _ = precision_recall_curve(
            self.exp.y_test, self.exp.test_preds
        )
        test_ap = average_precision_score(self.exp.y_test, self.exp.test_preds)

        fig, ax = plt.subplots()
        ax.plot(train_rc, train_pr, label=f"Train Average Precision = {train_ap:.2f}")
        ax.plot(test_rc, test_pr, label=f"Test Average Precision = {test_ap:.2f}")
        ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
        plt.legend()
        return fig


class MultiClassEvaluator:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.methods = ["print_classification_report", "plot_confusion_matrix"]

    def print_classification_report(self) -> pd.DataFrame:
        """Generates scikit-learn's classification report and reformats it as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing Precision, Recall, F1-Score and Support for all classes.
        """
        y_pred_test = self.exp.model.classes_[self.exp.test_preds.argmax(axis=1)]
        report = classification_report(
            y_true=self.exp.y_test, y_pred=y_pred_test, output_dict=True
        )
        return (
            pd.DataFrame(report)[self.exp.y_test.unique()]
            .T.sort_values("support", ascending=False)
            .astype({"support": int})[["f1-score", "precision", "recall", "support"]]
            .style.background_gradient(
                cmap="RdYlBu", axis="index", subset=["f1-score"], vmin=0, vmax=1,
            )
        )

    def plot_confusion_matrix(self) -> Figure:
        disp = plot_confusion_matrix(
            estimator=self.exp.model,
            X=self.exp.X_test,
            y_true=self.exp.y_test,
            xticks_rotation="vertical",
            cmap=plt.cm.Blues,
            labels=self.exp.y_test.value_counts().index,
        )
        return disp.figure_
