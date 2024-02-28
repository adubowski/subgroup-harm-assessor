import argparse
import logging
import time
from typing import Tuple, Union
import warnings

# Ignore UserWarning from shap
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from load import (
    add_bias,
    get_classifier,
    get_fairsd_result_set,
    load_data,
    get_shap_logloss,
)
from dash_app.config import APP_NAME
from dash_app.main import app
from dash_app.views.confusion_matrix import CMchart
from dash_app.views.menu import get_subgroup_dropdown_options
from plot import (
    get_data_distr_charts,
    get_data_table,
    get_feat_box,
    get_feat_shap_violin_plots,
    get_feat_table,
    get_sg_hist,
    plot_calibration_curve,
    plot_roc_curves,
)
from metrics import (
    Y_PRED_METRICS,
    get_qf_from_str,
    get_quality_metric_from_str,
    sort_quality_metrics_df,
)


def prepare_app(
    n_samples=0, dataset="adult", bias=False, train_split=True, model="rf"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Loads the data and trains the classifier

    Args:
        n_samples (int, optional): Number of samples to load. Defaults to 0.
        dataset (str, optional): Name of the dataset to load. Defaults to "adult".
        bias (Union[str, bool], optional): Type of bias to add to the dataset. Defaults to False.
            If set to "random", adds random noise to a random feature for a random subset of the data.
            If set to "swap", swaps the values of a random feature for a random subset of the data.
    Returns:
        Tuple: X_test, y_true_test, y_pred, y_pred_prob, shap_logloss_df, y_df
    """

    # Loading and training
    (
        X_test,
        y_true_train,
        y_true_test,
        onehot_X_train,
        onehot_X_test,
        cat_features,
    ) = load_data(n_samples=n_samples, dataset=dataset, train_split=train_split)

    random_subgroup = pd.Series(
        np.random.choice([True, False], size=len(X_test), p=[0.5, 0.5])
    )

    if bias:
        add_bias(bias, X_test, onehot_X_test, random_subgroup)

    classifier, y_pred_prob = get_classifier(
        onehot_X_train, y_true_train, onehot_X_test, model=model
    )

    shap_logloss_df = get_shap_logloss(
        classifier, onehot_X_test, y_true_test, X_test, cat_features
    )

    return X_test, y_true_test, y_pred_prob, shap_logloss_df, random_subgroup


def run_app(
    n_samples: int,
    dataset: str,
    bias: Union[str, bool] = False,
    random_subgroup=False,
    train_split=True,
    model="rf",
):
    """Runs the app with the given qf_metric"""
    use_random_subgroup = (
        random_subgroup or bias
    )  # When evaluating bias, we want to evaluate against a random subgroup
    start = time.time()
    (
        X_test,
        y_true_global_test,
        y_pred_prob_global,
        shap_logloss_df_global,
        random_subgroup_global,
    ) = prepare_app(
        n_samples=n_samples,
        dataset=dataset,
        bias=bias,
        train_split=train_split,
        model=model,
    )

    app.layout = html.Div(
        id="app-container",
        children=[
            dcc.Store(id="result-set-dict"),
            # Header
            dbc.Row(
                [
                    # Left column
                    dbc.Col(
                        id="left-column",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.H5(APP_NAME),
                                        style={
                                            "align-items": "center",
                                            "height": "fit-content",
                                            "white-space": "nowrap",
                                            "width": 4,
                                        },
                                    ),
                                    dbc.Col(
                                        html.H6(
                                            "1. Subgroup Discovery Metric: ",
                                        ),
                                        style={
                                            "display": "flex",
                                            "align-items": "center",
                                            "height": "fit-content",
                                            "justify-content": "right",
                                            "width": 4,
                                        },
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="fairness-metric-dropdown",
                                            options=[
                                                {
                                                    "label": "Equalized Odds Difference",
                                                    "value": "equalized_odds_diff",
                                                },
                                                {
                                                    "label": "Equalized Odds Ratio",
                                                    "value": "equalized_odds_ratio",
                                                },
                                                {
                                                    "label": "Avg Log Loss Difference",
                                                    "value": "average_log_loss_diff",
                                                },
                                                {
                                                    "label": "Avg Log Loss Ratio",
                                                    "value": "average_log_loss_ratio",
                                                },
                                                {
                                                    "label": "Brier Score Difference",
                                                    "value": "brier_score_diff",
                                                },
                                                {
                                                    "label": "Brier Score Ratio",
                                                    "value": "brier_score_ratio",
                                                },
                                                {
                                                    "label": "AUROC (ROC AUC) Difference",
                                                    "value": "auroc_diff",
                                                },
                                                {
                                                    "label": "AUROC (ROC AUC) Ratio",
                                                    "value": "auroc_ratio",
                                                },
                                                {
                                                    "label": "Miscalibration Difference",
                                                    "value": "miscalibration_diff",
                                                },
                                                {
                                                    "label": "Accuracy Difference",
                                                    "value": "acc_diff",
                                                },
                                                {
                                                    "label": "F1-score Difference",
                                                    "value": "f1_diff",
                                                },
                                                {
                                                    "label": "False Positive Rate Difference",
                                                    "value": "fpr_diff",
                                                },
                                                {
                                                    "label": "True Positive Rate Difference",
                                                    "value": "tpr_diff",
                                                },
                                                {
                                                    "label": "False Negative Rate Difference",
                                                    "value": "fnr_diff",
                                                },
                                            ],
                                            value="average_log_loss_diff",
                                        )
                                    ),
                                ]
                            ),
                        ],
                    ),
                    # Right column
                    dbc.Col(
                        id="right-column",
                        children=[
                            dbc.Row(
                                [
                                    # Insert center block
                                    dbc.Col(
                                        html.Center(
                                            html.H6("2. Select Subgroup: "),
                                        ),
                                        width=3,
                                        style={
                                            "align-items": "right",
                                            "height": "fit-content",
                                        },
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="subgroup-dropdown",
                                            options=[],
                                            # value=0,
                                            style={
                                                "align-items": "left",
                                                "height": "fit-content",
                                            },
                                        ),
                                        width=9,
                                        style={
                                            "align-items": "left",
                                            "height": "fit-content",
                                        },
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
                style={"height": "5vh"},
            ),
            dcc.Tabs(
                id="tabs",
                value="impact",
                children=[
                    dcc.Tab(
                        id="impact",
                        label="3. Misclassifications Overview",
                        value="impact",
                        children=[
                            # Split the tab into two columns
                            html.Div(
                                className="row",
                                children=[
                                    html.Div(
                                        className="six columns",
                                        children=[
                                            html.H6(
                                                "Full Dataset Baseline",
                                                style={
                                                    "border-bottom": "3px solid #d3d3d3"
                                                },
                                            ),
                                            # Update group description from callback
                                            dbc.Row(
                                                [
                                                    # Ensure the col is on the left from the confusion matrix
                                                    dbc.Col(
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    # [
                                                                    html.Div(
                                                                        id="simple-baseline-table",
                                                                        className="six-columns",
                                                                        children="Wait for the baseline to load...",
                                                                        style={
                                                                            "align-items": "center",
                                                                            "height": "fit-content",
                                                                        },
                                                                    ),
                                                                ),
                                                                dbc.Col(
                                                                    dcc.Graph(
                                                                        id="simple-baseline-conf",
                                                                        style={
                                                                            "align-items": "center",
                                                                            "height": "fit-content",
                                                                            "height": "20vh",
                                                                            "font-size": "0.8rem",
                                                                        },
                                                                    )
                                                                ),
                                                            ]
                                                        ),
                                                    ),
                                                ]
                                            ),
                                            html.Br(),
                                            # Add placeholder for graph
                                            dcc.Graph(id="simple-baseline-hist"),
                                            # Add slider for decision threshold
                                            html.H6(
                                                "Select decision threshold for the model:"
                                            ),
                                            dcc.Slider(
                                                0.1,
                                                0.9,
                                                0.1,
                                                value=0.5,
                                                id="simple-baseline-threshold-slider",
                                            ),
                                        ],
                                        style={
                                            "textAlign": "center",
                                            "margin-right": "0.5",
                                        },
                                    ),
                                    # html.Br(),
                                    html.Div(
                                        className="six columns",
                                        children=[
                                            html.H6(
                                                "(Dis)advantaged Subgroup",
                                                style={
                                                    "border-bottom": "3px solid #d3d3d3"
                                                },
                                            ),
                                            # Update group description from callback
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                id="simple-subgroup-col",
                                                                className="six-columns",
                                                                children="Select subgroup and wait for the visualizations to load. ",
                                                                style={
                                                                    "align-items": "center",
                                                                    "height": "fit-content",
                                                                },
                                                            ),
                                                        ]
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="simple-subgroup-conf",
                                                            # Update height and font size of confusion matrix
                                                            style={
                                                                "height": "20vh",
                                                                "font-size": "0.4rem",
                                                            },
                                                        )
                                                    ),
                                                ]
                                            ),
                                            html.Br(),
                                            dcc.Graph(id="simple-subgroup-hist"),
                                        ],
                                        style={
                                            "textAlign": "center",
                                            # "border-left": "3px solid #d3d3d3",
                                            "margin-left": "1",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="4. Performance and Calibration",
                        value="performance_tab",
                        children=[
                            # Split into two equal width columns with headers
                            html.Div(
                                className="row",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    dcc.Graph(
                                                        id="perf-roc",
                                                    ),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    # dcc.Graph(
                                                    #     id="perf-prc",
                                                    # ),
                                                    dcc.Graph(
                                                        id="calibration_curve",
                                                    ),
                                                    html.H6(
                                                        "Select number of bins for the calibration plot:"
                                                    ),
                                                    dcc.Slider(
                                                        4,
                                                        20,
                                                        4,
                                                        value=8,
                                                        id="calibration-slider",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    # Add whitespace below performance plots
                                    # dbc.Row([
                                    #     html.Br(),
                                    # ]),
                                    # dbc.Row([
                                    #     dcc.Graph(
                                    #         id="calibration_curve",
                                    #     ),
                                    #     html.H6("Select (max) number of bins for the calibration plot:"),
                                    #     dcc.Slider(
                                    #         5, 30, 5, value=10, id="calibration-slider"
                                    #     ),
                                    # ])
                                ],
                                style={"align-items": "center"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="5. Loss contributions per feature",
                        value="feature_contributions_tab",
                        children=[
                            dbc.Row(
                                [
                                    dcc.Graph(
                                        id="feat-bar",
                                        style={
                                            "align-items": "center",
                                            "height": "fit-content",
                                            # "height": "1000",
                                            "font-size": "0.8rem",
                                        },
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6(
                                                "Select significance level for the KS test (Data table rows in bold are significant at the selected level): "
                                            ),
                                            dcc.Slider(
                                                0.01,
                                                0.1,
                                                0.01,
                                                value=0.05,
                                                id="feat-alpha-slider",
                                            ),
                                            # Add slider for sensitivity of the test
                                            html.H6(
                                                "Select rounding level for the KS test: "
                                            ),
                                            dcc.Slider(
                                                0,
                                                7,
                                                1,
                                                value=1,
                                                id="feat-sensitivity-slider",
                                            ),
                                            html.H6(
                                                "Rounding level is the level at which SHAP values should be rounded because they are considered 'distinct'," +
                                                  "to avoid minor differences between distributions in the KS test."
                                            ),
                                        ]
                                    ),
                                    html.Br(),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                id="feat-table-col",
                                                className="six-columns",
                                                children="Data table for feature contributions. Select a subgroup to update the table.",
                                                style={
                                                    "align-items": "center",
                                                    "height": "fit-content",
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="6. Loss contributions per feature value",
                        value="feature_value_contributions_tab",
                        children=[
                            # Split into two equal width columns with headers
                            html.Div(
                                className="row",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    # Add header for feature selection
                                                    html.H6(
                                                        "Select feature for value contributions:"
                                                    ),
                                                    # Add dropdown for feature selection
                                                    dcc.Dropdown(
                                                        id="feat-val-feature-dropdown",
                                                        options=[
                                                            {"label": col, "value": col}
                                                            for col in X_test.columns
                                                        ],
                                                        value=X_test.columns[0],
                                                        style={
                                                            "align-items": "center",
                                                            "width": "50%",
                                                            "text-align": "center",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dbc.Col([]),
                                        ]
                                    ),
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            # Add a placeholder for the graph
                                            dcc.Graph(id="feat-val-bar"),
                                        ]
                                    ),
                                    html.Br(),
                                    dcc.Slider(
                                        2, 20, 2, value=8, id="feat-val-hist-slider"
                                    ),
                                ],
                                style={"align-items": "center"},
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="7. Class Imbalances",
                        value="data_tab",
                        children=[
                            # Split into two equal width columns with headers
                            html.Div(
                                className="row",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.H6(
                                                        "Select feature for distribution plots:"
                                                    ),
                                                    # Add dropdown for feature selection
                                                    dcc.Dropdown(
                                                        id="data-feature-dropdown",
                                                        options=[
                                                            {"label": col, "value": col}
                                                            for col in X_test.columns
                                                        ],
                                                        value=X_test.columns[0],
                                                        style={
                                                            "align-items": "center",
                                                            "width": "50%",
                                                            "text-align": "center",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            dbc.Col(
                                                [
                                                    # Add dropdown for percentage/absolute values
                                                    html.H6(
                                                        "Select aggregation method for distribution plots:"
                                                    ),
                                                    dcc.Dropdown(
                                                        id="data-agg-dropdown",
                                                        options=[
                                                            {
                                                                "label": "Percentage",
                                                                "value": "percentage",
                                                            },
                                                            {
                                                                "label": "Count",
                                                                "value": "count",
                                                            },
                                                        ],
                                                        value="percentage",
                                                        style={
                                                            "align-items": "center",
                                                            "width": "50%",
                                                            "text-align": "center",
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Br(),
                                    # dbc.Row([
                                    #     # Add a placeholder for the graph
                                    #     dcc.Graph(id="data-feature-dist-plot"),
                                    # ]),
                                    dbc.Row(
                                        [
                                            # Add a placeholder for the graph
                                            dcc.Graph(id="data-pos-class-dist-plot"),
                                        ]
                                    ),
                                    dbc.Row(
                                        [
                                            # Add a placeholder for the graph
                                            dcc.Graph(id="data-neg-class-dist-plot"),
                                        ]
                                    ),
                                    html.H6(
                                        "Select (max) number of bins (numerical features only):"
                                    ),
                                    dcc.Slider(
                                        5, 30, 5, value=20, id="data-hist-slider"
                                    ),
                                ],
                                style={"align-items": "center"},
                            ),
                        ],
                    ),
                ],
                # Set headers to bold font
                style={"font-weight": "bold"},
            ),
        ],
    )

    @app.callback(
        Output("simple-baseline-table", "children"),
        Output("simple-baseline-conf", "figure"),
        Output("simple-baseline-hist", "figure"),
        Output("subgroup-dropdown", "options"),
        Output("result-set-dict", "data"),
        Input("fairness-metric-dropdown", "value"),
        Input("simple-baseline-threshold-slider", "value"),
    )
    def get_baseline_stats_and_subgroups(metric, threshold):
        if not metric:
            raise PreventUpdate

        y_true = y_true_global_test.copy()
        y_pred_prob = y_pred_prob_global.copy()
        if metric in Y_PRED_METRICS:
            y_pred = (y_pred_prob >= threshold).astype(int)
        else:
            y_pred = y_pred_prob.copy()

        y_df = pd.DataFrame({"y_true": y_true, "probability": y_pred_prob})
        y_df["category"] = y_df.apply(
            lambda row: "TP"
            if row["y_true"] == 1 and row["probability"] >= threshold
            else "FP"
            if row["y_true"] == 0 and row["probability"] >= threshold
            else "FN"
            if row["y_true"] == 1 and row["probability"] < threshold
            else "TN",
            axis=1,
        )

        baseline_descr = "Full dataset baseline"

        # baseline_imp_bar = get_shap_barchart(shap_values_df, baseline_sg, "Mean contribution to model loss for the baseline (lower is better, features with non-negative importance are not helping the prediction)")
        baseline_data_table = get_data_table(
            baseline_descr,
            y_true,
            y_pred,
            y_pred_prob,
            qf_metric=metric,
            sg_feature=pd.Series([True] * y_true.shape[0]),
        )
        baseline_conf_mat = CMchart(
            "Confusion Matrix", y_true, (y_pred_prob >= threshold).astype(int)
        ).fig
        baseline_hist = get_sg_hist(
            y_df, title="Histogram of prediction probabilities on the full dataset"
        )

        if use_random_subgroup:
            sg_feature = random_subgroup_global.copy()
            # Replace the result_set_df with a synthetic random subgroup
            sg_y_pred = (
                y_pred[sg_feature]
                if metric in Y_PRED_METRICS
                else y_pred_prob[sg_feature]
            )
            sg_y_true = y_true[sg_feature]
            result_set_df = pd.DataFrame(
                {
                    "quality": [None],
                    "description": ["Random subgroup"],
                    "size": [sum(sg_feature)],
                    "proportion": [sum(sg_feature) / len(sg_feature)],
                    "metric_score": [
                        get_quality_metric_from_str(metric)(sg_y_true, sg_y_pred)
                    ],
                }
            )
            result_set_json = {
                "descriptions": ["Random subgroup"],
                "sg_features": [sg_feature.to_json()],
                "metric": metric,
            }
            return (
                baseline_data_table,
                baseline_conf_mat,
                baseline_hist,
                get_subgroup_dropdown_options(result_set_df, metric),
                result_set_json,
            )
        else:
            result_set = get_fairsd_result_set(
                X_test,
                y_true_global_test,
                y_pred,
                qf=get_qf_from_str(metric),
                # method="between_groups",
                method="to_overall",
                min_support_ratio=0.01,
                depth=1,
                max_support_ratio=0.9,  # To prevent finding majority subgroups
                logging_level=logging.INFO,
            )
            result_set_df = result_set.to_dataframe()
            # result_set_df['metric_score'] = 0.0
            # result_set_df['metric'] = metric
            metrics = []
            for idx in range(len(result_set_df)):
                # Add the metric value (e.g. Accuracy for acc_diff)
                description = result_set.get_description(idx)
                sg_feature = description.to_boolean_array(X_test)
                sg_y_pred = (
                    y_pred[sg_feature]
                    if metric in Y_PRED_METRICS
                    else y_pred_prob[sg_feature]
                )
                sg_y_true = y_true[sg_feature]
                metrics.append(
                    get_quality_metric_from_str(metric)(sg_y_true, sg_y_pred)
                )

            result_set_df["metric_score"] = metrics
            result_set_df = sort_quality_metrics_df(result_set_df, metric)
            # Get the result set json including the ordering from the sort (get the ordering
            return (
                baseline_data_table,
                baseline_conf_mat,
                baseline_hist,
                get_subgroup_dropdown_options(result_set_df, metric),
                result_set.to_json(
                    X_test, metric, result_set_df
                ),  # Store the result set representation in the data store
            )

    # Get feature value plot based on subgroup and feature selection
    @app.callback(
        Output("feat-val-bar", "figure"),
        Input("feat-val-feature-dropdown", "value"),
        Input("subgroup-dropdown", "value"),
        Input("result-set-dict", "data"),
        Input("feat-val-hist-slider", "value"),
    )
    def get_feat_val_bar(feature, subgroup, data, nbins):
        """Produces a bar chart or line plot with the feature value contributions for the selected subgroup"""
        if not feature:
            raise PreventUpdate
        if subgroup is None:
            raise PreventUpdate
        if not nbins:
            print("Error: No bins selected. This should not happen.")
            raise PreventUpdate

        if len(data["descriptions"]) == 0:
            print("Error: No subgroups found. This should not happen.")
            raise PreventUpdate

        description = data["descriptions"][subgroup]
        sg_feature = pd.read_json(data["sg_features"][subgroup], typ="series")
        return get_feat_shap_violin_plots(
            X_test.copy(),
            shap_logloss_df_global.copy(),
            sg_feature,
            feature,
            description,
            nbins=nbins,
        )

    # Get calibration plot based on subgroup and slider selection
    @app.callback(
        Output("calibration_curve", "figure"),
        Input("calibration-slider", "value"),
        Input("subgroup-dropdown", "value"),
        Input("result-set-dict", "data"),
    )
    def get_calibration_plot(slider_value, subgroup, data):
        """Produces a calibration plot for the selected subgroup"""
        if not slider_value:
            raise PreventUpdate
        if subgroup is None:
            raise PreventUpdate
        if len(data["sg_features"]) == 0:
            print("Error: No subgroups found. This should not happen.")
            raise PreventUpdate
        sg_feature = pd.read_json(data["sg_features"][subgroup], typ="series")
        y_true = y_true_global_test.copy()
        y_pred_prob = y_pred_prob_global.copy()

        return plot_calibration_curve(
            y_true, y_pred_prob, sg_feature, n_bins=slider_value
        )

    # Get data distributions based on subgroup selection
    @app.callback(
        # Output("data-feature-dist-plot", "figure"),
        Output("data-pos-class-dist-plot", "figure"),
        Output("data-neg-class-dist-plot", "figure"),
        Input("data-feature-dropdown", "value"),
        Input("data-agg-dropdown", "value"),
        Input("subgroup-dropdown", "value"),
        Input("result-set-dict", "data"),
        Input("data-hist-slider", "value"),
    )
    def get_data_feat_distr(feature, agg, subgroup, data, bins):
        """Produces a bar chart or line plot with the data feature values counts for the selected subgroup"""
        if not feature:
            raise PreventUpdate
        if not agg:
            raise PreventUpdate
        if subgroup is None:
            raise PreventUpdate
        if not bins:
            logging.error("Error: No bins selected. This should not happen.")
            raise PreventUpdate

        try:
            description = data["descriptions"][subgroup]
            sg_feature = pd.read_json(data["sg_features"][subgroup], typ="series")
        except IndexError:
            print("Subgroup not found. This should not happen.")
            raise PreventUpdate

        y_true = y_true_global_test.copy()
        return get_data_distr_charts(
            X_test, y_true, sg_feature, feature, description, bins, agg
        )

    # Get feat-table-col
    @app.callback(
        Output("feat-table-col", "children"),
        Input("subgroup-dropdown", "value"),
        Input("result-set-dict", "data"),
        Input("feat-alpha-slider", "value"),
        Input("feat-sensitivity-slider", "value"),
    )
    def get_feat_table_col(subgroup, data, alpha, sensitivity):
        """Returns the feature contributions table for the selected subgroup"""
        if subgroup is None:
            raise PreventUpdate
        if not alpha:
            alpha = 0.05
        if len(data["descriptions"]) == 0:
            print("Error: No subgroups found. This should not happen.")
            raise PreventUpdate

        sg_feature = pd.read_json(data["sg_features"][subgroup], typ="series")
        shap_df = shap_logloss_df_global.copy()
        return get_feat_table(
            shap_values_df=shap_df,
            sg_feature=sg_feature,
            sensitivity=sensitivity,
            alpha=alpha,
        )

    # Get plots based on the subgroup selection
    @app.callback(
        Output("simple-subgroup-col", "children"),
        Output("simple-subgroup-conf", "figure"),
        Output("simple-subgroup-hist", "figure"),
        Output("perf-roc", "figure"),
        Output("feat-bar", "figure"),
        Input("result-set-dict", "data"),
        Input("subgroup-dropdown", "value"),
        Input("simple-baseline-threshold-slider", "value"),
    )
    def get_subgroup_stats(data, subgroup, threshold):
        """Returns the group description and updates the charts of the selected subgroup"""
        if subgroup is None:
            raise PreventUpdate
        if len(data["descriptions"]) == 0:
            print("Error: No subgroups found. This should not happen.")
            raise PreventUpdate

        sg_feature = pd.read_json(data["sg_features"][subgroup], typ="series")
        description = data["descriptions"][subgroup]
        subgroup_description = str(description).replace(" ", "")
        metric = data["metric"]

        y_true = y_true_global_test.copy()

        y_pred_prob = y_pred_prob_global.copy()
        y_pred = (y_pred_prob >= threshold).astype(int)

        y_df = pd.DataFrame({"y_true": y_true, "probability": y_pred_prob})
        y_df["category"] = y_df.apply(
            lambda row: "TP"
            if row["y_true"] == 1 and row["probability"] >= threshold
            else "FP"
            if row["y_true"] == 0 and row["probability"] >= threshold
            else "FN"
            if row["y_true"] == 1 and row["probability"] < threshold
            else "TN",
            axis=1,
        )
        shap_values_df = shap_logloss_df_global.copy()

        sg_hist = get_sg_hist(y_df[sg_feature])

        sg_data_table = get_data_table(
            subgroup_description,
            y_true,
            y_pred,
            y_pred_prob,
            qf_metric=metric,
            sg_feature=sg_feature,
        )

        roc_fig = plot_roc_curves(
            y_true, y_pred_prob, sg_feature, title="ROC for subgroup and baseline"
        )

        sg_conf_mat = CMchart(
            "Confusion Matrix", y_true_global_test[sg_feature], y_pred[sg_feature]
        ).fig

        # Get a bar chart with the feature importances for sg and baseline
        feat_box = get_feat_box(shap_values_df, sg_feature=sg_feature)

        return (sg_data_table, sg_conf_mat, sg_hist, roc_fig, feat_box)

    print("App startup time (s): ", time.time() - start)
    app.run_server(debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=0,
        help="Number of samples to use for the app. Use 0 to load the entire dataset.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="adult",
        help="Dataset to be used in the evaluation. Available options are: 'adult', 'credit_g', 'heloc'",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=str,
        default=False,
        help="Type of bias to add to the dataset",
    )
    parser.add_argument(
        "-r",
        "--random_subgroup",
        action="store_true",
        default=False,
        help="Flag whether to use a random subgroup for evaluation",
    )
    parser.add_argument(
        "-s",
        "--train_split",
        default=True,
        help="Flag whether to split the number of samples selected into train and test. Only test data is then used for visualizations",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="rf",
        help="Model to use for the evaluation. Available options are: 'rf', 'dt', 'xgb'",
    )
    args = parser.parse_args()
    # bias = args.bias if args.bias in ("random", "mean", "swap", "bin", "binning") else False
    train_split = False if args.train_split in ("False", "false", False, "F") else True
    run_app(
        args.n_samples,
        args.dataset,
        args.bias,
        args.random_subgroup,
        train_split,
        args.model,
    )
