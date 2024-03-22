import logging
import time
from typing import List

from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from utils import (
    combine_one_hot,
    combine_all_one_hot_shap_logloss,
)

from fairsd import fairsd
from fairsd.fairsd.algorithms import ResultSet


def import_dataset(dataset: str, sensitive_features: List[str] = None):
    """Import the dataset from OpenML and preprocess it.
    Args:
        dataset (str): Dataset to be imported. Supported datasets: adult, german_credit, heloc, credit.
        sensitive_features (list, optional): Sensitive features to be used in the dataset. Defaults to None.
    """

    print("Loading data...")
    cols_to_drop = []
    # Import dataset
    if dataset == "adult":
        dataset_id = 1590
        target = ">50K"
        cols_to_drop = ["fnlwgt", "education-num", "sex", "race", "native-country"]
        if sensitive_features is not None:
            cols_to_drop = [col for col in cols_to_drop if col not in sensitive_features]
    elif dataset in ("credit_g", "german", "german_credit"):
        dataset_id = 31
        target = "good"
        cols_to_drop = ["personal_status", "other_parties", "residence_since", "foreign_worker"]
        if sensitive_features is not None:
            cols_to_drop = [col for col in cols_to_drop if col not in sensitive_features]
    elif dataset == "heloc":
        dataset_id = 45023
        target = "1"
    elif dataset == "credit":
        dataset_id = 43978
        target = 1
    else:
        raise NotImplementedError(
            "Only the following datasets are supported now: adult, german_credit, heloc, credit."
        )

    d = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    X = d.data
    if sensitive_features is not None:
        incorrect_sensitive_features = [
            col for col in sensitive_features if col not in X.columns
        ]
        if incorrect_sensitive_features:
            raise ValueError(
                f"Sensitive features {incorrect_sensitive_features} not found in the dataset."
            )

    X = X.drop(cols_to_drop, axis=1, errors="ignore")

    # Fill missing values - "missing" for categorical, mean for numerical
    for col in X.columns:
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.add_categories("missing").fillna("missing")
        elif X[col].dtype.name == "object":
            X[col] = X[col].fillna("missing")
        else:
            X[col] = X[col].fillna(X[col].mean())

    # Get target as 1/0
    y_true = (d.target == target) * 1
    return X, y_true


def load_data(
    dataset="adult",
    n_samples=0,
    train_split=True,
    sensitive_features: List[str] = None,
):
    """Load data from UCI Adult dataset.
    Args:
        dataset (str): Dataset to be loaded, currently only Adult and German Credit datasets are supported
        n_samples (int, optional): Number of samples to load. Select 0 to load all.
        train_split (bool, optional): Flag whether to split the number of selected data samples into train and test
        sensitive_features (list, optional): Sensitive features to be used in the dataset. Defaults to None.
    """
    X, y_true = import_dataset(dataset, sensitive_features)

    # Get categorical feature names
    cat_features = X.select_dtypes(include=["category"]).columns

    # One-hot encoding
    onehot_X = pd.get_dummies(X) * 1

    if n_samples and X.shape[0] >= n_samples:
        # Load only n examples
        X = X.iloc[:n_samples]
        y_true = y_true.iloc[:n_samples]
        onehot_X = onehot_X.iloc[:n_samples]

    if train_split:
        (
            X_train,
            X_test,
            y_true_train,
            y_true_test,
            onehot_X_train,
            onehot_X_test,
        ) = train_test_split(X, y_true, onehot_X, test_size=0.3, random_state=0)
        # Reset indices
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_true_train = y_true_train.reset_index(drop=True)
        y_true_test = y_true_test.reset_index(drop=True)
        onehot_X_train.reset_index(inplace=True, drop=True)
        onehot_X_test.reset_index(inplace=True, drop=True)

        X_test.reset_index(inplace=True, drop=True)
        y_true_test.reset_index(inplace=True, drop=True)

        return (
            X_test,
            y_true_train,
            y_true_test,
            onehot_X_train,
            onehot_X_test,
            cat_features,
        )

    else:
        return X, y_true, y_true, onehot_X, onehot_X, cat_features


def add_bias(
    bias: str, X_test: pd.DataFrame, onehot_X_test: pd.DataFrame, subgroup: pd.Series
) -> pd.Series:
    """Add bias to the dataset."""

    if bias in ("random", "noise"):
        feature = "capital-gain"
        # Add random noise to the subset
        std_val = X_test[feature].std()
        mean_val = X_test[feature].mean()
        X_test.loc[subgroup, feature] += np.random.normal(
            mean_val, std_val, sum(subgroup)
        )
        onehot_X_test.loc[subgroup, feature] += np.random.normal(
            mean_val, std_val, sum(subgroup)
        )
    elif bias == "mean":
        feature = "age"
        # Add the mean of the feature to the subset
        X_test.loc[subgroup, feature] = X_test[feature].mean()
        onehot_X_test.loc[subgroup, feature] = onehot_X_test[feature].mean()
    elif bias == "median":
        feature = "age"
        # Add the median of the feature to the subset
        X_test.loc[subgroup, feature] = X_test[feature].median()
        onehot_X_test.loc[subgroup, feature] = onehot_X_test[feature].median()
    elif bias in ("bin", "binning"):
        feature = "age"
        # Add the binning of the feature to the subset
        X_test.loc[subgroup, feature] = X_test[subgroup].apply(
            lambda x: x[feature] // 20 * 20, axis=1
        )
        onehot_X_test.loc[subgroup, feature] = onehot_X_test[subgroup].apply(
            lambda x: x // 20 * 20, axis=1
        )
    elif bias == "sum_std":
        feature = "age"
        std_val = X_test[feature].std()
        # Add the standard deviation of the feature to the subset
        X_test.loc[subgroup, feature] = X_test[subgroup].apply(
            lambda x: x[feature] + std_val, axis=1
        )
        onehot_X_test.loc[subgroup, feature] = onehot_X_test[feature].sum()

    elif bias == "swap":
        # Swap all values of the feature to another value in the same column
        # feature = "education"
        # value_selected = "Doctorate"
        feature = "marital-status"
        value_selected = "Married-civ-spouse"
        X_test.loc[subgroup, feature] = value_selected
        # Onehot_X_test is one hot encoded so we need to swap the entire column for the feature
        onehot_X_test.loc[subgroup, onehot_X_test.columns.str.startswith(feature)] = 0
        onehot_X_test.loc[subgroup, [feature + "_" + value_selected]] = 1
    else:
        raise ValueError(
            f"Bias method '{bias}' not supported. Supported methods: random, mean, median, bin, sum_std, swap."
        )
    print(
        f"Added bias to the dataset by method: {bias}. Feature {feature} was affected. Size of the subset impacted: {sum(subgroup)}."
    )


def get_classifier(
    onehot_X_train: pd.DataFrame,
    y_true_train: pd.Series,
    onehot_X_test: pd.DataFrame,
    with_names=True,
    model="rf",
):
    """Get a decision tree classifier for the given dataset.

    Args:
        onehot_X_train: one-hot encoded features dataframe used for training
        y_true_train: true labels for training
        onehot_X_test: one-hot encoded features dataframe used for testing
    """
    if model == "rf":
        classifier = RandomForestClassifier(
            n_estimators=20, max_depth=8, random_state=0
        )
    elif model == "dt":
        classifier = DecisionTreeClassifier(min_samples_leaf=30, max_depth=8)
    elif model == "xgb":
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
    else:
        raise ValueError(f"Model {model} not supported. Supported models: rf, dt, xgb.")
        
    if not with_names:
        onehot_X_train = onehot_X_train.values
        onehot_X_test = onehot_X_test.values

    print("Training the classifier...")
    classifier.fit(onehot_X_train, y_true_train)

    y_pred_prob = classifier.predict_proba(onehot_X_test)

    return classifier, y_pred_prob[:, 1]


def combine_shap_one_hot(shap_values, X_columns, cat_features):
    # Combine one-hot encoded cat_features
    non_cat_features = [col for col in X_columns if col not in cat_features]
    for cat_feat in cat_features:
        # Get column masks for each cat feature
        col_masks = [
            col.startswith(cat_feat) and col not in non_cat_features
            for col in shap_values.feature_names
        ]

        shap_values = combine_one_hot(
            shap_values, cat_feat, col_masks, return_original=False
        )
    return shap_values


def get_shap_values(classifier, d_train, X, cat_features, combine_cat_features=True):
    """Get shap values for a given classifier and dataset.
    Combines one-hot encoded categorical features into original features."""
    # Producing shap values
    explainer = shap.TreeExplainer(classifier)
    # TODO: Add interation values
    # shap_interaction = explainer.shap_interaction_values(X)

    shap_values = explainer(d_train)

    if combine_cat_features:
        shap_values = combine_shap_one_hot(shap_values, X.columns, cat_features)
    return shap_values


def get_shap_logloss(
    classifier, d_train, y_true, X, cat_features, combine_cat_features=True
):
    """Get shap values of the model log loss for a given classifier and dataset.
    Combines one-hot encoded categorical features into original features."""
    explainer_bg_100 = shap.TreeExplainer(
        classifier,
        shap.sample(d_train, 100),
        feature_perturbation="interventional",
        model_output="log_loss",
    )
    shap_values_logloss_all = explainer_bg_100.shap_values(d_train, y_true)
    if len(shap_values_logloss_all) == 2:
        shap_values_logloss_all = shap_values_logloss_all[1]
    print("Shap values log loss shape: ", shap_values_logloss_all.shape)
    print(len(d_train.columns))
    shap_logloss_df = pd.DataFrame(shap_values_logloss_all, columns=d_train.columns)
    if combine_cat_features:
        shap_logloss_df = combine_all_one_hot_shap_logloss(
            shap_logloss_df, X.columns, cat_features
        )
    return shap_logloss_df


def get_fairsd_result_set(
    X,
    y_true,
    y_pred,
    qf="equalized_odds_difference",
    method="to_overall",
    min_quality=0.01,
    depth=1,
    min_support=100,
    result_set_size=30,
    sensitive_features=None,
    **kwargs,
) -> ResultSet:
    """Get result set from fairsd DSSD task

    Args:
        X (pd.DataFrame): Dataset
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
        qf (str, optional): Quality function. Defaults to "equalized_odds_difference".
        depth (int, optional): Depth of subgroup discovery. Defaults to 2.
        min_support (int, optional): Minimum support. Defaults to 30.
        result_set_size (int, optional): Size of result set. Defaults to 10.
        kwargs: Additional arguments to pass to fairsd.SubgroupDiscoveryTask,
            including result_set_ratio and logging_level (as defined by logging module)
    """
    if sensitive_features is not None:
        X = X[sensitive_features].copy()
    task = fairsd.SubgroupDiscoveryTask(
        X,
        y_true,
        y_pred,
        qf=qf,
        depth=depth,
        result_set_size=result_set_size,
        min_quality=min_quality,
        min_support=min_support,
        **kwargs,
    )
    if "logging_level" in kwargs:
        logging_level = kwargs["logging_level"]
    else:
        logging_level = logging.WARNING

    if logging_level < logging.WARNING:
        # Only print if logging level is lower than WARNING
        print(f"Running DSSD...")
        start = time.time()
    if method == "to_overall":
        result_set = fairsd.DSSD(beam_width=30).execute(task, method="to_overall")
    else:
        result_set = fairsd.DSSD(beam_width=30).execute(task)
    if logging_level < logging.WARNING:
        print(f"DSSD Done! Total time: {time.time() - start}")
    return result_set
