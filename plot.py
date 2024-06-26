import numpy as np
import pandas as pd
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from metrics import (
    Y_PRED_METRICS,
    get_quality_metric_from_str,
    get_name_from_metric_str,
    miscalibration_score,
)
from scipy.stats import ks_2samp #, wasserstein_distance

COLS_TO_SHOW = [
    "quality",
    "size",
    "description",
    "auc_diff",
    "f1_diff",
    "exp_js_div",
    "exp_mi",
    "exp_ks_test_p_val",
    "stat_diff_exp",
]


def plot_pr_curves(y_true, y_pred, y_pred_prob, sg_feature, title=None):
    """Plots PRC curves for the subgroup and the baseline of the subgroup"""

    baseline_size = len(y_true)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    auprc_baseline = average_precision_score(y_true, y_pred)
    # Plot with plotly
    fig = go.Figure()
    # Sort the precision, recall and thresholds according to recall
    precision, recall, thresholds = zip(
        *sorted(zip(precision, recall, thresholds), key=lambda x: x[0])
    )
    fig.add_trace(
        go.Scatter(
            x=precision,
            y=recall,
            name="Baseline (area = %0.2f, n = %d)" % (auprc_baseline, baseline_size),
            mode="lines+markers",
            customdata=thresholds,
            hovertemplate="Precision: %{x}<br>Recall: %{y}<br>Threshold: %{customdata}",
        )
    )

    subgroup_size = sum(sg_feature)
    sg_precision, sg_recall, sg_tresholds = precision_recall_curve(
        y_true[sg_feature], y_pred_prob[sg_feature]
    )
    auprc_subgroup = average_precision_score(y_true[sg_feature], y_pred[sg_feature])
    sg_precision, sg_recall, sg_tresholds = zip(
        *sorted(zip(sg_precision, sg_recall, sg_tresholds), key=lambda x: x[0])
    )
    fig.add_trace(
        go.Scatter(
            x=sg_precision,
            y=sg_recall,
            name="Subgroup (area = %0.2f, n = %d)" % (auprc_subgroup, subgroup_size),
            mode="lines+markers",
            customdata=sg_tresholds,
            hovertemplate="Precision: %{x}<br>Recall: %{y}<br>Threshold: %{customdata}",
        )
    )
    # Mark axes
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    # Update title
    if title:
        fig.update_layout(title=title)
    # Update height
    fig.update_layout(height=550)
    return fig


def plot_roc_curves(y_true, y_pred_prob, sg_feature, title=None):
    """Plots ROC curves for the subgroup and the baseline of the subgroup"""

    fig = go.Figure()

    # Plot ROC curve for baseline
    baseline_size = len(y_true)
    baseline_fpr, baseline_tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc_group2 = auc(baseline_fpr, baseline_tpr)
    fig.add_trace(
        go.Scatter(
            x=baseline_fpr,
            y=baseline_tpr,
            name="Baseline (area = %0.3f, n = %d)" % (roc_auc_group2, baseline_size),
            mode="lines+markers",
            customdata=thresholds,
            hovertemplate="False Positive Rate: %{x}<br>True Positive Rate: %{y}<br>Threshold: %{customdata}",
        )
    )

    # Plot ROC curve for subgroup
    group_size1 = sum(sg_feature)
    sg_fpr, sg_tpr, sg_thresholds = roc_curve(
        y_true[sg_feature], y_pred_prob[sg_feature]
    )
    auroc_subgroup = auc(sg_fpr, sg_tpr)

    fig.add_trace(
        go.Scatter(
            x=sg_fpr,
            y=sg_tpr,
            name="Subgroup (area = %0.3f, n = %d)" % (auroc_subgroup, group_size1),
            mode="lines+markers",
            customdata=sg_thresholds,
            hovertemplate="False Positive Rate: %{x}<br>True Positive Rate: %{y}<br>Threshold: %{customdata}",
        )
    )
    # Mark axes
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")

    # Update title
    if title:
        fig.update_layout(title=title)
    # Update height
    fig.update_layout(height=550)
    return fig


def plot_calibration_curve(
    y_true, y_pred_prob, sg_feature, n_bins=10, strategy="uniform"
):
    """Plots calibration curve for a classifier for group and its opposite"""

    fig = go.Figure()
    for group in ["Baseline", "Subgroup"]:
        group_filter = (
            sg_feature if group == "Subgroup" else pd.Series([True] * len(y_true))
        )
        cal_curve = calibration_curve(
            y_true=y_true[group_filter],
            y_prob=y_pred_prob[group_filter],
            n_bins=n_bins,
            strategy=strategy,
        )
        # Write the calibration plot with plotly
        fig.add_trace(
            go.Scatter(
                x=cal_curve[1],
                y=cal_curve[0],
                name=group
                + "<br> (miscalibration score = %0.3f, n = %d)"
                % (
                    miscalibration_score(
                        y_true[group_filter], y_pred_prob[group_filter], n_bins=n_bins
                    ),
                    sum(group_filter),
                ),
                # Add number of datapoints included at each point
                customdata=group_filter.sum() * range(1, n_bins + 1) // n_bins,
                mode="lines+markers",
                hovertemplate="Mean predicted probability: %{x}<br>Fraction of positives: %{y}<br>Subgroup size: %{customdata}",
            )
        )
    # Add perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash')
        )
    )
    fig.update_layout(title="Calibration curves for the selected subgroup and baseline")
    fig.update_xaxes(title_text="Mean predicted probability")
    fig.update_yaxes(title_text="Fraction of positives")
    # Update height
    fig.update_layout(height=550)
    return fig


def get_sg_hist(y_df_local, categories=["TN", "FN", "TP", "FP"], title=None):
    """Returns a histogram of the predictions for the subgroup

    Args:
        y_df_local (pd.DataFrame): A dataframe with the true labels and the predictions
    """
    y_df_local = y_df_local[y_df_local["category"].isin(categories)]
    sg_hist = px.histogram(
        y_df_local,
        x="probability",
        color="category",
        hover_data=y_df_local.columns,
        category_orders={"category": categories},
    )
    # Hide TN and TP by default
    # sg_hist.for_each_trace(lambda t: t.update(visible="legendonly") if t.name in ["TN", "TP"] else t)

    sg_hist.update_xaxes(range=[0, 1])
    sg_hist.update_traces(
        xbins=dict(
            start=0,
            end=1,
            size=0.1,
        ),
    )
    if title is None:
        title = "Histogram of prediction probabilities for the selected subgroup"
    sg_hist.update_layout(
        title_text=title, legend_title_text="", modebar_remove=["zoom", "pan"]
    )
    sg_hist.layout.xaxis.fixedrange = True
    sg_hist.layout.yaxis.fixedrange = True
    # Update histogram height
    sg_hist.update_layout(height=550)
    return sg_hist


def get_data_table(
    subgroup_description, y_true, y_pred, y_pred_prob, qf_metric, sg_feature, n_bins=10
):
    """Generates a data table with the subgroup description and the subgroup size"""

    # tpr = true_positive_score(y_true[sg_feature], y_pred[sg_feature]).round(3)
    # fpr = false_positive_score(y_true[sg_feature], y_pred[sg_feature]).round(3)
    auroc = roc_auc_score(y_true[sg_feature], y_pred_prob[sg_feature]).round(3)
    # auprc = average_precision_score(y_true[sg_feature], y_pred[sg_feature]).round(3)
    cal_score = miscalibration_score(
        y_true[sg_feature], y_pred_prob[sg_feature], n_bins=n_bins
    ).round(3)
    # avg_loss = log_loss(y_true[sg_feature], y_pred_prob[sg_feature]).round(3)
    brier_score = np.mean((y_true[sg_feature] - y_pred_prob[sg_feature]) ** 2).round(3)

    metric_name = get_name_from_metric_str(qf_metric)

    if qf_metric in Y_PRED_METRICS:
        quality_score = get_quality_metric_from_str(qf_metric)(
            y_true[sg_feature], y_pred[sg_feature]
        )
    else:
        quality_score = get_quality_metric_from_str(qf_metric)(
            y_true[sg_feature], y_pred_prob[sg_feature]
        )
    # fp = sum((y_true[sg_feature] == 0) & (y_pred[sg_feature] == 1))
    # fn = sum((y_true[sg_feature] == 1) & (y_pred[sg_feature] == 0))
    
    # Generate a data table with the subgroup description
    table_content = {
        "Statistic": [
            "Description",
            "Size",
            "AUROC",
            "Miscalibration score",
            "Brier score",
            metric_name
        ],
        "Value": [
            subgroup_description,
            sg_feature.sum(),
            auroc,
            cal_score,
            brier_score,
            quality_score
        ],
    }
    # if metric_name != "Average Log Loss":
    #     table_content["Statistic"].append(metric_name)
    #     table_content["Value"].append(quality_score)

    df = pd.DataFrame(table_content)
    data_table = dash_table.DataTable(
        style_table={"overflowX": "auto"},
        style_data={"whiteSpace": "normal", "height": "auto"},
        data=df.to_dict("records"),
        style_cell_conditional=[
            {"if": {"column_id": "Feature"}, "width": "35%"},
        ],
    )

    return data_table


def get_data_distr_charts(
    X, y_pred, sg_feature, feature, description, nbins=20, agg="percentage"
):
    """For positive and negative labels and predictions, returns a figure with the histogram distribution across 
    feature values of the selected feature in the subgroup and the baseline"""

    # Get the data distribution for the feature values of the selected feature in the subgroup and the baseline
    class_plot = get_data_distr_chart(
        X, sg_feature, feature, description, nbins, agg
    )
    class_plot.update_layout(
        title="Data distribution for "
        + feature
        + f" in the subgroup ({description}) and the baseline"
    )
    # Get the predictions distribution for the feature values of the selected feature in the subgroup and the baseline
    pred_plot = get_data_distr_chart(
        y_pred, sg_feature, feature, description, nbins, agg
    )
    pred_plot.update_layout(
        title="Predictions distribution for "
        + feature
        + f" in the subgroup ({description}) and the baseline"
    )
    if agg == "percentage":
        # Update yaxis range to 1
        class_plot.update_layout(yaxis=dict(range=[0, 100]))
        pred_plot.update_layout(yaxis=dict(range=[0, 100]))
    
    return class_plot, pred_plot


def get_data_distr_chart(
    X, sg_feature, feature, description, nbins=20, agg="percentage"
):
    """Returns a figure with the data distribution for the feature values of the selected feature in the subgroup and the baseline"""

    if type(X) == pd.DataFrame and feature in X.columns:
        X = X[feature]
    else:
        X = pd.Series(X).astype("str")
    X_sg = X[sg_feature].copy()

    fig = go.Figure()
    # Add trace for baseline
    fig.add_trace(
        go.Histogram(
            x=X,
            name="Baseline",
            histnorm="percent" if agg == "percentage" else "",
            nbinsx=nbins,
        )
    )

    # Add trace for subgroup
    fig.add_trace(
        go.Histogram(
            x=X_sg,
            name="Subgroup",
            histnorm="percent" if agg == "percentage" else "",
            nbinsx=nbins,
        )
    )

    # Update layout
    fig.update_layout(
        title="Data distribution for "
        + feature
        + f" in the subgroup ({description}) and the baseline",
        xaxis_title=feature,
        yaxis_title=agg.capitalize() + " of data in respective group",
    )
    # Update yaxis range to max value
    # max_y = max(
    #     max(fig.data[0].y), # FIXME: How to get the max value of the histogram object?
    #     max(fig.data[1].y)
    # )
    # fig.update_layout(yaxis=dict(range=[0, max_y]))

    return fig


def get_feat_val_violin_plots(shap_values_df, sg_feature=None):
    """Returns a violin plot for the features SHAP values in the subgroup and the baseline"""
    shap_values_df = transform_box_data(shap_values_df, sg_feature)

    # Create the fig and add hoverdata of confidence intervals
    fig = px.violin(
        shap_values_df,
        x="feature",
        y="shap_value",
        color="group",
        points="all",
        title="Feature contributions to model loss for subgroup and baseline. <br> "
        + "The lower the value, the higher its informativeness.",
        hover_data=shap_values_df.columns,
        height=600,
    )
 
 
    # Update layout
    fig.update_layout(
        yaxis_title="SHAP log loss value",
        xaxis_title="Feature",
        violinmode="group",
    )
    # Update height
    fig.update_layout(height=600)
    return fig


def get_feat_val_violin_plot(X, shap_df, sg_feature, feature, description, nbins=20):
    """Returns a figure with a violin plot for the feature value SHAP contributions in the subgroup and the baseline"""
    feature_type = "categorical"
    orig_df = X.copy()
    # If data is continuous, we need to bin it
    if X[feature].dtype in [np.float64, np.int64]:
        X[feature] = pd.cut(X[feature], bins=nbins)
        bins = X[feature].cat.categories.astype(str)
        sorted_bins = sorted(
            bins, key=lambda x: float(x.split(",")[0].strip("(").strip(" ").strip("]"))
        )
        X[feature] = X[feature].astype(str)
        feature_type = "continuous"

    # Merge the shap values with the feature values such that we can plot the violin plot of shap values per feature value
    concat_df = pd.concat([shap_df[feature], X[feature]], axis=1)
    concat_df.columns = ["SHAP", feature]
    
    # Create a violin plot for the feature values
    fig = go.Figure()

    # Add trace for baseline
    fig.add_trace(
        go.Violin(
            x=concat_df[feature],
            y=concat_df["SHAP"],
            name="Baseline",
            box_visible=True,
            meanline_visible=True,
            points="all",
            text=orig_df.apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),
            hoverinfo="y+text",
        )
    )

    # Add trace for subgroup
    fig.add_trace(
        go.Violin(
            x=concat_df[feature][sg_feature],
            y=concat_df["SHAP"][sg_feature],
            name="Subgroup",
            box_visible=True,
            meanline_visible=True,
            points="all",
            text=orig_df[sg_feature].apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),            hoverinfo="y+text",
        )
    )
    slider_note = (
        "Use slider below to adjust the (max) number of bins for the violin plot"
        if feature_type == "continuous"
        else "When the selected feature is categorical, slider changes do not affect the plot."
    )
    # Update layout
    fig.update_layout(
        title="Feature distribution for "
        + feature
        + f" in the subgroup ({description}) and the baseline <br>"
        + "The lower the feature's values, the higher its informativeness",
        xaxis_title=f"Feature values of {feature} <br> Feature type: {feature_type}; {slider_note}",
        yaxis_title="SHAP log loss value",
        violinmode="group",
        # yaxis=dict(range=[-0.4, 0.4])
    )
    # If feature is continuous, we need to sort the bins
    if feature_type == "continuous":
        fig.update_xaxes(categoryorder="array", categoryarray=sorted_bins)
    else:
        fig.update_xaxes(categoryorder="category ascending")
    # Update height
    fig.update_layout(height=700)
    return fig


def get_feat_val_box(X, shap_df, sg_feature, feature, description, nbins=20):
    """Returns a figure with a box plot for the feature value SHAP contributions in the subgroup and the baseline"""
    feature_type = "categorical"
    orig_df = X.copy()
    # If data is continuous, we need to bin it
    if X[feature].dtype in [np.float64, np.int64]:
        X[feature] = pd.cut(X[feature], bins=nbins)
        bins = X[feature].cat.categories.astype(str)
        sorted_bins = sorted(
            bins, key=lambda x: float(x.split(",")[0].strip("(").strip(" ").strip("]"))
        )
        X[feature] = X[feature].astype(str)
        feature_type = "continuous"

    # Merge the shap values with the feature values such that we can plot the violin plot of shap values per feature value
    concat_df = pd.concat([shap_df[feature], X[feature]], axis=1)
    concat_df.columns = ["SHAP", feature]
    
    # Create a violin plot for the feature values
    fig = go.Figure()

    # Add trace for baseline
    fig.add_trace(
        go.Box(
            x=concat_df[feature],
            y=concat_df["SHAP"],
            name="Baseline",
            boxpoints="all",
            text=orig_df.apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),
            hoverinfo="y+text",
        )
    )

    # Add trace for subgroup
    fig.add_trace(
        go.Box(
            x=concat_df[feature][sg_feature],
            y=concat_df["SHAP"][sg_feature],
            name="Subgroup",
            boxpoints="all",
            text=orig_df[sg_feature].apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),
            hoverinfo="y+text",
        )
    )
    slider_note = (
        "Use slider below to adjust the (max) number of bins for the violin plot"
        if feature_type == "continuous"
        else "When the selected feature is categorical, slider changes do not affect the plot."
    )
    # Update layout
    fig.update_layout(
        title="Feature distribution for "
        + feature
        + f" in the subgroup ({description}) and the baseline <br>"
        + "The lower the feature's values, the higher its informativeness",
        xaxis_title=f"Feature"
        + f" values of {feature} <br> Feature type: {feature_type}; {slider_note}",
        yaxis_title="SHAP log loss value",
    )
    # If feature is continuous, we need to sort the bins
    if feature_type == "continuous":
        fig.update_xaxes(categoryorder="array", categoryarray=sorted_bins)
    else:
        fig.update_xaxes(categoryorder="category ascending")
    # Update height
    fig.update_layout(height=700)
    return fig


def get_feat_val_bar(X, shap_df, sg_feature, feature, description, nbins=20, agg="mean"):
    """Returns a figure with a bar plot for the feature value SHAP contributions in the subgroup and the baseline"""
    feature_type = "categorical"
    orig_df = X.copy()
    # If data is continuous, we need to bin it
    if X[feature].dtype in [np.float64, np.int64]:
        X[feature] = pd.cut(X[feature], bins=nbins)
        bins = X[feature].cat.categories.astype(str)
        sorted_bins = sorted(
            bins, key=lambda x: float(x.split(",")[0].strip("(").strip(" ").strip("]"))
        )
        X[feature] = X[feature].astype(str)
        feature_type = "continuous"

    # Merge the shap values with the feature values such that we can plot the violin plot of shap values per feature value
    concat_df = pd.concat([shap_df[feature], X[feature]], axis=1)
    concat_df.columns = ["SHAP", feature]
    
    # Create a violin plot for the feature values
    fig = go.Figure()

    if agg == "sum_weighted":
        # Calculate the weighted sum of the SHAP values by using a sum divided over the total group size
        baseline_df = concat_df.groupby(feature)["SHAP"].agg("sum") / len(concat_df)
        sg_df = concat_df[sg_feature].groupby(feature)["SHAP"].agg("sum") / len(concat_df[sg_feature])
        title = "Weighted sum"
    else:
        baseline_df = concat_df.groupby(feature)["SHAP"].agg(agg)
        sg_df = concat_df[sg_feature].groupby(feature)["SHAP"].agg(agg)
        title = agg.capitalize()
    
    # Add trace for baseline
    fig.add_trace(
        go.Bar(
            x=baseline_df.index,
            y=baseline_df.values,
            name="Baseline",
            text=orig_df.apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),
            hoverinfo="y+text",
        )
    )
    # Add trace for subgroup
    fig.add_trace(
        go.Bar(
            x=sg_df.index,
            y=sg_df.values,
            name="Subgroup",
            text=orig_df[sg_feature].apply(lambda row: ' '.join(f'{i}:{v}' for i, v in row.items()), axis=1),
            hoverinfo="y+text",
        )
    )
    slider_note = (
        "Use slider below to adjust the (max) number of bins for the violin plot"
        if feature_type == "continuous"
        else "When the selected feature is categorical, slider changes do not affect the plot."
    )
    # Update layout
    fig.update_layout(
        title="SHAP feature value loss contributions for "
        + feature
        + f" in the subgroup ({description}) and the baseline <br>"
        + "The lower the feature's values, the higher its informativeness",
        xaxis_title=f"Feature"
        + f" values of {feature} <br> Feature type: {feature_type}; {slider_note}",
        yaxis_title=title + " of SHAP log loss values",
    )
    # If feature is continuous, we need to sort the bins
    if feature_type == "continuous":
        fig.update_xaxes(categoryorder="array", categoryarray=sorted_bins)
    else:
        fig.update_xaxes(categoryorder="category ascending")
    # Update height
    fig.update_layout(height=700)
    return fig


def get_feat_bar(shap_values_df, sg_feature, agg, error_bars=False) -> go.Figure:
    """Returns a figure with the feature contributions to the model loss

    Args:
        shap_values_df (pd.DataFrame): The shap values dataframe
        sg_feature (pd.Series): The subgroup feature
        agg (str): The aggregation method (mean, median, sum, mean_diff)
        error_bars (bool): Whether to add error bars to the plot
    Returns:
        go.Figure: The figure with the feature contributions to the model loss
    """

    sg_shap_values_df = shap_values_df[sg_feature]
    if agg == "mean":
        # Get the mean absolute shap value for each feature
        shap_values_df_agg = shap_values_df.mean(numeric_only=True)
        sg_shap_values_agg = sg_shap_values_df.mean(numeric_only=True)
    elif agg == "median":
        # Get the median absolute shap value for each feature
        shap_values_df_agg = shap_values_df.median(numeric_only=True)
        sg_shap_values_agg = sg_shap_values_df.median(numeric_only=True)
    elif agg == "sum":
        # Get the sum of absolute shap value for each feature
        shap_values_df_agg = shap_values_df.sum(numeric_only=True)
        sg_shap_values_agg = sg_shap_values_df.sum(numeric_only=True)
    elif agg == "mean_diff":
        # Get the difference in mean shap value for each feature
        shap_values_df_agg = sg_shap_values_df.mean(numeric_only=True) - shap_values_df.mean(numeric_only=True)
        # Then produce only a single bar plot with the difference
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=shap_values_df_agg.index,
                y=shap_values_df_agg,
                name="Difference (Subgroup - Baseline)",
            )
        )
        fig.update_layout(
            title="Difference in mean of the SHAP feature contributions to model loss (Subgroup - Baseline)",
            xaxis_title="Feature",
            yaxis_title="Difference in SHAP contributions to loss",
        )
        return fig
    
    # Combine the two dataframes
    shap_values_df_agg = pd.concat([shap_values_df_agg, sg_shap_values_agg], axis=1)
    shap_values_df_agg.columns = ["Baseline", "Subgroup"]

    # Create the fig and add hoverdata of confidence intervals
    fig = go.Figure()
    ytitle = agg.capitalize() + " SHAP log loss value - feature contribution to loss"

    if error_bars:
        ytitle += " <br> With standard deviation error bars"
        fig.add_trace(
            go.Bar(
                x=shap_values_df_agg.index,
                y=shap_values_df_agg["Baseline"],
                name="Baseline",
                customdata=shap_values_df.std(axis=0, numeric_only=True),
                hovertemplate="Feature: %{x}<br>Baseline: %{y}<br>Standard deviation: %{customdata}",
                error_y=dict(
                    type="data",
                    array=shap_values_df.std(axis=0, numeric_only=True),
                    visible=True,
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=shap_values_df_agg.index,
                y=shap_values_df_agg["Subgroup"],
                name="Subgroup",
                customdata=sg_shap_values_df.std(axis=0, numeric_only=True),
                hovertemplate="Feature: %{x}<br>Subgroup: %{y}<br>Standard deviation: %{customdata}",
                error_y=dict(
                    type="data",
                    array=sg_shap_values_df.std(axis=0, numeric_only=True),
                    visible=True,
                ),
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=shap_values_df_agg.index,
                y=shap_values_df_agg["Baseline"],
                name="Baseline",
            )
        )
        fig.add_trace(
            go.Bar(
                x=shap_values_df_agg.index,
                y=shap_values_df_agg["Subgroup"],
                name="Subgroup",
            )
        )

    # Update the fig
    fig.update_layout(
        barmode="group",
        yaxis_tickangle=-45,
        title="Feature contributions to model loss for subgroup and baseline. <br> "
        + "The lower the value, the higher its informativeness.",
        yaxis_title=ytitle,
        xaxis_title="Feature",
        height=600,
    )
    # Order x axis by the absolute value of the difference between the subgroup and the baseline
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=shap_values_df_agg.abs().diff(axis=1).sort_values(by="Subgroup").index,
    )
    # Turn y labels
    fig.update_layout(xaxis_tickangle=-25)
    return fig


def get_feat_box(shap_values_df, sg_feature=None) -> go.Figure:
    """Returns a figure with the feature contributions to the model loss

    Args:
        shap_values_df (pd.DataFrame): The shap values dataframe
        sg_feature (pd.Series): The subgroup feature
        title (str): The title of the figure
    Returns:
        go.Figure: The figure with the feature contributions to the model loss
    """
    shap_values_df = transform_box_data(shap_values_df, sg_feature)

    # Create the fig and add hoverdata of confidence intervals
    fig = px.box(
        shap_values_df,
        x="feature",
        y="shap_value",
        color="group",
        points=False,
        title="Feature contributions to model loss for subgroup and baseline. <br> "
        + "The lower the value, the higher its informativeness.",
        hover_data=shap_values_df.columns,
        height=600,
    )

    # Update the fig
    fig.update_layout(
        yaxis_title="SHAP log loss value - feature contribution to loss",
        xaxis_title="Feature",
    )
    fig.update_traces(boxmean=True)

    return fig


def transform_box_data(shap_values_df, sg_feature):
    """Transforms the shap values dataframe to be suitable for a box plot"""
    shap_values_df = shap_values_df.drop(columns="group", errors="ignore")

    if sg_feature is not None:
        sg_shap_values_df = shap_values_df[sg_feature]

        # Put all shap values of different features in a single column with feature names as a new column
        sg_shap_values_df = sg_shap_values_df.reset_index()
        sg_shap_values_df = sg_shap_values_df.melt(
            id_vars="index", var_name="feature", value_name="shap_value"
        )
        sg_shap_values_df["group"] = "Subgroup"

    # Put all shap values of different features in a single column with feature names as a new column
    shap_values_df = shap_values_df.reset_index()
    shap_values_df = shap_values_df.melt(
        id_vars="index", var_name="feature", value_name="shap_value"
    )
    shap_values_df["group"] = "Baseline"

    if sg_feature is not None:
        # Combine the two dataframes
        shap_values_df = pd.concat([shap_values_df, sg_shap_values_df], axis=0)
    # Drop column index
    shap_values_df = shap_values_df.drop(columns="index")
    return shap_values_df


def get_feat_table(shap_values_df, sg_feature, sensitivity=4, alpha=0.05):
    """Returns a data table with the feature contributions to the model loss summary and tests for significance"""

    sg_shap_values_df = shap_values_df[sg_feature]

    # Get the mean absolute shap value for each feature
    shap_values_df_mean = shap_values_df.mean(numeric_only=True).round(5)
    sg_shap_values_mean = sg_shap_values_df.mean(numeric_only=True).round(5)
    shap_values_df_mean = pd.concat([shap_values_df_mean, sg_shap_values_mean], axis=1)
    shap_values_df_mean.columns = ["Baseline", "Subgroup"]

    # Get the standard deviation of the shap values
    shap_values_df_std = shap_values_df.std(numeric_only=True).round(5)
    sg_shap_values_df_std = sg_shap_values_df.std(numeric_only=True).round(5)
    shap_values_df_std = pd.concat([shap_values_df_std, sg_shap_values_df_std], axis=1)
    shap_values_df_std.columns = ["Baseline", "Subgroup"]

    # Get the p-value of the shap values
    shap_values_df_p = pd.DataFrame(
        index=shap_values_df_mean.index, columns=["KS_p_value"]
    )

    # Round shap values to 5 decimal places
    shap_values_df = shap_values_df.round(sensitivity)
    sg_shap_values_df = sg_shap_values_df.round(sensitivity)

    for feature in shap_values_df_mean.index:
        # Run KS test
        statistic, p_value = ks_2samp(
            shap_values_df[feature], sg_shap_values_df[feature]
        )
        shap_values_df_p.loc[feature, "KS_p_value"] = p_value.round(6)
        shap_values_df_p.loc[feature, "KS_statistic"] = statistic

        # Calculate Wasserstein distance
        # wasserstein_dist = wasserstein_distance(
        #     shap_values_df[feature], sg_shap_values_df[feature]
        # )
        # shap_values_df_p.loc[feature, "Wasserstein_distance"] = wasserstein_dist.round(6)

    shap_values_df_p = shap_values_df_p.round(6)

    # Merge the dataframes
    df = shap_values_df_mean.merge(
        shap_values_df_std, left_index=True, right_index=True
    )
    df = df.merge(shap_values_df_p, left_index=True, right_index=True)
    df = df.reset_index()
    df.columns = [
        "Feature",
        "Baseline_avg",
        "Subgroup_avg",
        "Baseline_std",
        "Subgroup_std",
        "KS p-value",
        "KS statistic",
        # "Wasserstein dist",
    ]

    df["Cohen's d"] = (df["Subgroup_avg"] - df["Baseline_avg"]) / np.sqrt(
        (df["Baseline_std"] ** 2 + df["Subgroup_std"] ** 2) / 2
    )
    df["Cohen's d"] = df["Cohen's d"].round(5)

    # Order df rows based on the p-value and the mean
    df = df.sort_values(by=["KS statistic"], ascending=[False])

    # Merge avg and std columns
    df["Baseline (mean ± std)"] = (
        df["Baseline_avg"].astype(str) + " ± " + df["Baseline_std"].astype(str)
    )
    df["Subgroup (mean ± std)"] = (
        df["Subgroup_avg"].astype(str) + " ± " + df["Subgroup_std"].astype(str)
    )
    df = df.drop(
        columns=["Baseline_avg", "Subgroup_avg", "Baseline_std", "Subgroup_std"]
    )

    # Reorder columns
    df = df[
        [
            "Feature",
            "Baseline (mean ± std)",
            "Subgroup (mean ± std)",
            "Cohen's d",
            "KS statistic",
            "KS p-value",
            # "Wasserstein dist",
        ]
    ]

    # Generate a data table with the feature contributions to the model loss
    data_table = dash_table.DataTable(
        style_table={"overflowX": "auto"},
        style_data={"whiteSpace": "normal", "height": "auto"},
        data=df.to_dict("records"),
        # Format p-values in bold font if below 0.05
        style_data_conditional=[
            {
                "if": {"filter_query": "{KS p-value} < " + str(alpha)},
                "fontWeight": "bold",
            },
            # Change background of cells to gray in the column "Cohen's d"
            {
                "if": {"column_id": "Cohen's d"},
                "backgroundColor": "#f9f9f9",
            },
        ],
    )
    return data_table
