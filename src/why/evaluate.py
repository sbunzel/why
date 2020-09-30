from typing import Optional
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
from sklearn import metrics

from .explainer import Explainer

__all__ = ["BinaryClassEvaluator", "MultiClassEvaluator"]


class BinaryClassEvaluator:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.methods = ["plot_class_performance", "plot_precision_recall_curve"]

    def plot_precision_recall_curve(self, **kwargs) -> Figure:
        """Calculates and plots the precision recall curve for as set of training and test set predictions.
        
        Returns:
            Figure: Matplotlib figure of train and test precision recall curves.
        """
        train_pr, train_rc, _ = metrics.precision_recall_curve(
            self.exp.y_train, self.exp.train_preds
        )
        train_ap = metrics.average_precision_score(
            self.exp.y_train, self.exp.train_preds
        )
        test_pr, test_rc, _ = metrics.precision_recall_curve(
            self.exp.y_test, self.exp.test_preds
        )
        test_ap = metrics.average_precision_score(self.exp.y_test, self.exp.test_preds)

        fig, ax = plt.subplots()
        ax.plot(train_rc, train_pr, label=f"Train Average Precision = {train_ap:.2f}")
        ax.plot(test_rc, test_pr, label=f"Test Average Precision = {test_ap:.2f}")
        ax.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
        plt.legend()
        return fig

    def class_performance(self, threshold: float, **kwargs) -> pd.DataFrame:
        report = metrics.classification_report(
            y_true=self.exp.y_test,
            y_pred=self.exp.test_preds > threshold,
            output_dict=True,
        )
        df = (
            pd.DataFrame(report)[[str(c) for c in self.exp.y_test.unique()]]
            .T.astype({"support": int})[["f1-score", "precision", "recall", "support"]]
            .reset_index()
            .rename(columns={"index": "target"})
        )
        return df

    def plot_class_performance(self, threshold: float, **kwargs) -> alt.Chart:
        perf_df = self.class_performance(threshold=threshold)
        return plot_class_performance(perf_df=perf_df)


class MultiClassEvaluator:
    def __init__(self, exp: Explainer) -> None:
        self.exp = exp
        self.methods = [
            "plot_class_performance",
            "plot_confusion_matrix",
        ]
        self.y_pred_test = self.exp.model.classes_[self.exp.test_preds.argmax(axis=1)]

    def base_classification_report(self) -> pd.DataFrame:
        """Generates scikit-learn's classification report and reformats it as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing Precision, Recall, F1-Score and Support for all classes.
        """
        report = metrics.classification_report(
            y_true=self.exp.y_test,
            y_pred=self.y_pred_test,
            output_dict=True,
            zero_division=0,
        )
        return (
            pd.DataFrame(report)[self.exp.y_test.unique()]
            .T.sort_values("support", ascending=False)
            .astype({"support": int})[["f1-score", "precision", "recall", "support"]]
            .reset_index()
            .rename(columns={"index": "target"})
        )

    def classification_report(self, **kwargs):
        return self.base_classification_report().style.background_gradient(
            cmap="RdYlBu", axis="index", subset=["f1-score"], vmin=0, vmax=1,
        )

    def plot_confusion_matrix(self, **kwargs) -> Figure:
        disp = metrics.plot_confusion_matrix(
            estimator=self.exp.model,
            X=self.exp.X_test,
            y_true=self.exp.y_test,
            xticks_rotation="vertical",
            cmap=plt.cm.Blues,
            labels=self.exp.y_test.value_counts().index,
        )
        return disp.figure_

    def plot_class_performance(
        self, target: Optional[str] = None, **kwargs
    ) -> alt.Chart:
        perf_df = self.base_classification_report()
        if not target == "Show all":
            perf_df = perf_df.query(f"target == '{target}'")
        return plot_class_performance(perf_df=perf_df)


evaluators = {
    "binary_classification": BinaryClassEvaluator,
    "multi_class_classification": MultiClassEvaluator,
}


def plot_class_performance(perf_df: pd.DataFrame) -> alt.Chart:
    plot_df = perf_df.drop(columns="support").melt(
        id_vars=["target"], var_name="metric", value_name="score"
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=(0, 1))),
            y=alt.Y("target:N", title=""),
            row=alt.Row("metric:N", title=""),
            color=alt.Color(
                "metric:N", legend=None, scale=alt.Scale(scheme="tableau10")
            ),
        )
        .properties(width="container", title="Classification Report")
        .configure_title(fontSize=14, offset=10, orient="top", anchor="middle")
    )
