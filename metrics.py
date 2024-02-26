import numpy as np
import pandas as pd
from typing import Callable, Union
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, accuracy_score, f1_score, average_precision_score
from sklearn.calibration import calibration_curve
from fairlearn.metrics import make_derived_metric

true_positive_score = lambda y_true, y_pred: (y_true & y_pred).sum() / y_true.sum()
false_positive_score = lambda y_true, y_pred: ((1-y_true) & y_pred).sum() / ((1-y_true)).sum()
false_negative_score = lambda y_true, y_pred: 1 - true_positive_score(y_true, y_pred)
Y_PRED_METRICS = ("auprc_diff", "auprc_ratio", "accuracy_diff", "accuracy_ratio", "f1_diff", "f1_ratio", "equalized_odds_diff", "equalized_odds_ratio")

def average_log_loss_score(y_true, y_pred):
    """Average log loss function. """
    return log_loss(y_true, y_pred)


def miscalibration_score(y_true, y_pred, n_bins=10):
    """Miscalibration score. Calibration is the difference between the predicted and the true probability of the positive class."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))


def get_qf_from_str(metric: str, transform: str = "difference") -> Union[Callable[[pd.Series, pd.Series, pd.Series], float], str]:
    """Get the quality function from a string.
    
    Args:
        metric (str): Name of the metric. If None, the default metric is used.
        transform (str): Type of the metric. Can be "difference", "ratio" or other fairlearn supported transforms
    
    Returns: the quality function according to the selected metric - a defined function 
    or a string (in case of equalized odds difference as it's already defined in fairsd)"""
    # Preprocess metric string. If it ends with "diff" or "ratio", set transform accordingly
    if metric.split("_")[-1] == "diff":
        transform = "difference"
    elif metric.split("_")[-1] == "ratio":
        transform = "ratio"

    metric = trim_transform_from_str(metric).lower()

    if metric in ("equalized_odds", "eo", "eo_diff"):
        qf = "equalized_odds_difference" if transform == "difference" else "equalized_odds_ratio"
    elif metric in ("brier_score", "brier_score_loss"):
        qf = make_derived_metric(metric=brier_score_loss, transform=transform)
    elif metric in ("log_loss", "loss", "total_loss"):
        qf = make_derived_metric(metric=log_loss, transform=transform)
    elif metric in ("accuracy", "accuracy_score", "acc"):
        qf = make_derived_metric(metric=accuracy_score, transform=transform)
    elif metric in ("f1", "f1_score"):
        qf = make_derived_metric(metric=f1_score, transform=transform)
    elif metric in ("al", "average_loss", "average_log_loss"):
        qf = make_derived_metric(metric=average_log_loss_score, transform=transform)
    elif metric in ("roc_auc", "auroc", "auc_roc", "roc_auc_score"):
        qf = make_derived_metric(metric=roc_auc_score, transform=transform)
    elif metric in ("miscalibration", "miscal", "cal", "calibration"):
        qf = make_derived_metric(metric=miscalibration_score, transform=transform)
    elif metric in ("auprc", "pr_auc", "precision_recall_auc", "average_precision_score"):
        qf = make_derived_metric(metric=average_precision_score, transform=transform)
    elif metric in ("false_positive_rate", "fpr"):
        qf = make_derived_metric(metric=false_positive_score, transform=transform)
    elif metric in ("true_positive_rate", "tpr"):
        qf = make_derived_metric(metric=true_positive_score, transform=transform)
    elif metric in ("fnr", "false_negative_rate"):
        qf = make_derived_metric(metric=false_negative_score, transform=transform)
    else:
        raise ValueError(f"Metric: {metric} not supported. "
                         "Metric must be one of the following: "
                         "equalized_odds, brier_score_loss, log_loss, accuracy_score, average_loss, "
                         "roc_auc_diff, miscalibration_diff, auprc_diff, fpr_diff, tpr_diff")
    
    return qf

def get_name_from_metric_str(metric: str) -> str:
    """Get the name of the metric from a string nicely formatted."""
    metric = trim_transform_from_str(metric)
    if metric in ("equalized_odds", "eo"):
        return "TPR / FPR"
    # Split words and Capitalize the first letters
    return " ".join([word.upper() if word in ("auprc, auroc", "auc", "roc", "prc", "tpr", "fpr", "fnr") else word.capitalize() for word in metric.split("_") ])
    
def trim_transform_from_str(metric: str) -> str:
    """Trim the transform from a string."""
    if metric.split("_")[-1] == "diff" or metric.split("_")[-1] == "ratio":
        metric = "_".join(metric.split("_")[:-1])
    return metric

def get_quality_metric_from_str(metric: str) -> str:
    """Get the quality metric from a string."""

    if metric.split("_")[-1] == "diff" or metric.split("_")[-1] == "ratio":
        metric = "_".join(metric.split("_")[:-1]).lower()

    if metric in ("equalized_odds", "eo"):
        # Get max of tpr and fpr 
        return lambda y_true, y_pred: str(true_positive_score(y_true, y_pred).round(3)) + "; " + str(false_positive_score(y_true, y_pred).round(3))
    elif metric in ("brier_score", "brier_score_loss"):
        quality_metric = brier_score_loss
    elif metric in ("log_loss", "loss", "total_loss"):
        quality_metric = log_loss
    elif metric in ("accuracy", "accuracy_score", "acc"):
        quality_metric = accuracy_score
    elif metric in ("f1", "f1_score"):
        quality_metric = f1_score
    elif metric in ("al", "average_loss", "average_log_loss"):
        quality_metric = average_log_loss_score
    elif metric in ("roc_auc", "auroc", "auc_roc", "roc_auc_score"):
        quality_metric = roc_auc_score
    elif metric in ("miscalibration", "miscal"):
        quality_metric = miscalibration_score
    elif metric in ("auprc", "pr_auc", "precision_recall_auc", "average_precision_score"):
        quality_metric = average_precision_score
    elif metric in ("false_positive_rate", "fpr"):
        quality_metric = false_positive_score
    elif metric in ("true_positive_rate", "tpr"):
        quality_metric = true_positive_score
    elif metric in ("fnr", "false_negative_rate"):
        quality_metric = false_negative_score
    else:
        raise ValueError(f"Metric: {metric} not supported. "
                         "Metric must be one of the following: "                         "equalized_odds, brier_score_loss, log_loss, accuracy_score, "
                         "average_loss, roc_auc_diff, miscalibration_diff, auprc_diff, fpr_diff, tpr_diff")
    
    return lambda y_true, y_pred: quality_metric(y_true, y_pred).round(3)

def sort_quality_metrics_df(result_set_df: pd.DataFrame, quality_metric: str) -> pd.DataFrame:
    """Sort the result set dataframe by the quality metric."""
    # If quality_metric ends with ratio
    if quality_metric.split("_")[-1] == "ratio":
        # if ratios are below 1.0, the metric is more significant the lower it is so we sort in ascending order
        if result_set_df["metric_score"].max() < 1.0:
            # Sort the result_set_df in ascending order based on the metric_score
            result_set_df = result_set_df.sort_values(by="metric_score", ascending=True)
        # if ratios are above 1.0, the metric is more significant the higher it is so we sort in descending order
        else:
            # Sort the result_set_df in descending order based on the metric_score
            result_set_df = result_set_df.sort_values(by="metric_score", ascending=False)
    elif quality_metric.split("_")[-1] in ("difference", "diff"):
        # If metric is a loss (lower is better)
        if "loss" in quality_metric or "miscal" in quality_metric or "fpr" in quality_metric or "fnr" in quality_metric:
            # Sort the result_set_df in descending order based on the metric_score
            result_set_df = result_set_df.sort_values(by="metric_score", ascending=False)
        # If max differences are below one, we are talking about difference in ratios, so we should show the results in ascending order
        else:
            # Sort the result_set_df in ascending order based on the metric_score
            result_set_df = result_set_df.sort_values(by="metric_score", ascending=True)
    else:
        raise ValueError("Metric must be either a difference or a ratio! Provided metric:", quality_metric)
    
    return result_set_df