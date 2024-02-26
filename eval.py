import logging
import time
from typing import List
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from utils import (
    combine_one_hot,
    combine_all_one_hot_shap_logloss,
)

from fairsd import fairsd
from fairsd.fairsd.algorithms import ResultSet


def load_data(dataset="adult", n_samples=0, cols_to_drop=["fnlwgt", "education-num"], train_split=True):
    """Load data from UCI Adult dataset.
    Args:
        dataset (str): Dataset to be loaded, currently only Adult and German Credit datasets are supported
        n_samples (int, optional): Number of samples to load. Select 0 to load all.
        cols_to_drop (list, optional): Columns to drop from dataset. Defaults to ["fnlwgt", "education-num"].
        train_split (bool, optional): Flag whether to split the number of selected data samples into train and test 
    """
    # Import dataset
    if dataset == "adult":
        dataset_id = 1590
        target = ">50K"
    elif dataset in ("credit_g", "german", "german_credit"):
        dataset_id = 31 # 31 for German credit, 1590 for Adult
        target = "good"
    elif dataset == "heloc":
        dataset_id = 45023
        target = "1"
    elif dataset == "credit":
        dataset_id = 43978
        target = 1
    else:
        raise NotImplementedError("Only adult and german credit datasets are supported now")
    
    d = fetch_openml(
        data_id=dataset_id, 
        as_frame=True, parser='auto')
    X = d.data

    # Get categorical feature names
    cat_features = X.select_dtypes(include=["category"]).columns

    X = X.drop(cols_to_drop, axis=1, errors="ignore")

    # Get target as 1/0
    y_true = (d.target == target) * 1
    
    # One-hot encoding
    onehot_X = pd.get_dummies(X) * 1

    if n_samples and X.shape[0] >= n_samples:
        # Load only n examples
        X = X.iloc[:n_samples]
        y_true = y_true.iloc[:n_samples]
        onehot_X = onehot_X.iloc[:n_samples]
    
    if train_split:
        X_train, X_test, y_true_train, y_true_test, onehot_X_train, onehot_X_test = train_test_split(
            X, 
            y_true, 
            onehot_X,
            test_size=0.3, 
            random_state=42
        )
        # Reset indices
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_true_train = y_true_train.reset_index(drop=True)
        y_true_test = y_true_test.reset_index(drop=True)
        onehot_X_train.reset_index(inplace=True, drop=True)
        onehot_X_test.reset_index(inplace=True, drop=True)

        X_test.reset_index(inplace=True, drop=True)
        y_true_test.reset_index(inplace=True, drop=True)

        return X_test, y_true_train, y_true_test, onehot_X_train, onehot_X_test, cat_features
    
    else:
        return X, y_true, y_true, onehot_X, onehot_X, cat_features



def add_bias(bias: str, X_test: pd.DataFrame, onehot_X_test: pd.DataFrame, subgroup: pd.Series) -> pd.Series:

    if bias == "random":
        # Currently we only do the eval on capital-gain
        feature = 'capital-gain'
        
        std_val = X_test[feature].std()
        mean_val = X_test[feature].mean()

        X_test.loc[subgroup, feature] += np.random.normal(mean_val, std_val, len(subgroup))
    elif bias == "mean":
        feature = 'capital-gain'
        # Select a random subset of the data
        # Add the mean of the feature to the subset
        X_test.loc[subgroup, feature] = X_test[feature].mean()
        onehot_X_test.loc[subgroup, feature] = onehot_X_test[feature].mean()
    elif bias == "swap":
        feature = 'marital-status'
        # Swap the values of the feature to another value in the same column, onehot_X_test is one hot encoded so we need to swap the entire column for the feature
        value_selected = np.random.choice(X_test[feature].unique(), sum(subgroup))
        value_selected = "Divorced"
        X_test.loc[subgroup, feature] = value_selected
        # Set all columns to 0
        print("Adding bias to onehot_X_test")
        print(onehot_X_test.columns.str.startswith(feature))
        onehot_X_test.loc[subgroup, onehot_X_test.columns.str.startswith(feature)] = 0 # FIXME: IF this resets the column - how are we left with so many counts of the feature value?
        onehot_X_test.loc[subgroup, [feature + "_" + value_selected]] = 1
        print("Bias added to onehot_X_test")
    else:
        raise ValueError(f"bias should be either False, 'random', or 'swap', not {bias}")
    print(f"Added bias to the dataset by {bias} method. Feature {feature} was affected. Subset impacted: {len(subgroup)}.")


def get_classifier(onehot_X_train: pd.DataFrame, y_true_train: pd.Series, onehot_X_test: pd.DataFrame, with_names=True):
    """Get a decision tree classifier for the given dataset.

    Args:
        onehot_X_train: one-hot encoded features dataframe used for training
        y_true_train: 
        onehot_X_test: 
    """
    # Training the classifier
    # TODO: Switch to RandomForestClassifier and fix bugs
    # classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)

    classifier = DecisionTreeClassifier(min_samples_leaf=30, max_depth=8)
    if not with_names:
        onehot_X_train = onehot_X_train.values
        onehot_X_test = onehot_X_test.values
    
    classifier.fit(onehot_X_train, y_true_train)

    # Producing y_pred
    y_pred = classifier.predict(onehot_X_test)
    y_pred_prob = classifier.predict_proba(onehot_X_test)

    return classifier, y_pred, y_pred_prob[:, 1]


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


def get_shap_logloss(classifier, d_train, y_true, X, cat_features, combine_cat_features=True):
    """Get shap values of the model log loss for a given classifier and dataset.
    Combines one-hot encoded categorical features into original features."""
    explainer_bg_100 = shap.TreeExplainer(
        classifier, 
        shap.sample(d_train, 100),
        feature_perturbation="interventional", 
        model_output="log_loss"
    )
    shap_values_logloss_all = explainer_bg_100.shap_values(d_train, y_true)
    shap_logloss_df = pd.DataFrame(shap_values_logloss_all[1], columns=d_train.columns)
    if combine_cat_features:
        shap_logloss_df = combine_all_one_hot_shap_logloss(shap_logloss_df, X.columns, cat_features)
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
    result_set_size=20,
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
    task = fairsd.SubgroupDiscoveryTask(
        X,
        y_true,
        y_pred,
        qf=qf,
        depth=depth,
        result_set_size=result_set_size,
        min_quality=min_quality,
        min_support=min_support,
        **kwargs
    )
    if 'logging_level' in kwargs:
        logging_level = kwargs['logging_level']
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
