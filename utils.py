import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import make_derived_metric
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

roc_auc_diff = make_derived_metric(metric=roc_auc_score, transform="difference")
f1_score_diff = make_derived_metric(metric=f1_score, transform="difference")
precision_diff = make_derived_metric(metric=precision_score, transform="difference")
recall_diff = make_derived_metric(metric=recall_score, transform="difference")


def combine_all_one_hot_shap_logloss(
    sage_values_df: pd.DataFrame, target_features, cat_features
):
    """Combine all one hot encoded features into parent features"""
    sage_values_df = sage_values_df.copy()
    # Combine one hot encoded features with sage values
    non_cat_features = [col for col in target_features if col not in cat_features]
    for cat_feat in cat_features:
        # Get column masks for each cat feature
        col_mask = [
            col.startswith(cat_feat) and col not in non_cat_features
            for col in sage_values_df.columns
        ]
        # Sum columns from col_mask
        sage_values_df[cat_feat] = sage_values_df.loc[:, col_mask].sum(axis=1)
        # Add cat_feat to col_mask
        if len(col_mask) < sage_values_df.shape[1]:
            col_mask.append(False)
        # Drop columns from col_mask
        sage_values_df.drop(sage_values_df.columns[col_mask], axis=1, inplace=True)
    return sage_values_df


def combine_one_hot(shap_values, name, mask, return_original=True):
    """Combines one-hot-encoded features into a single feature

    Args:
        shap_values: an Explanation object
        name: name of new feature
        mask: bool array same lenght as features

    This function assumes that shap_values[:, mask] make up a one-hot-encoded feature
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype="object")[mask]

    sv_name = shap.Explanation(
        shap_values.values[:, mask],
        feature_names=list(mask_col_names),
        data=shap_values.data[:, mask],
        base_values=shap_values.base_values,
        display_data=shap_values.display_data,
        instance_names=shap_values.instance_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )

    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)

    svdata = np.concatenate(
        [shap_values.data[:, ~mask], new_data.reshape(-1, 1)], axis=1
    )

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate(
        [svdd, mask_col_names[new_data].reshape(-1, 1)], axis=1
    )

    new_values = sv_name.values.sum(axis=1)

    # Reshape new_values to match the dims of shap_values.values
    svvalues = np.concatenate(
        [shap_values.values[:, ~mask], new_values.reshape(-1, 1, 2)], axis=1
    )

    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(
        svvalues,
        base_values=shap_values.base_values,
        data=svdata,
        display_data=svdisplay_data,
        instance_names=shap_values.instance_names,
        feature_names=svfeature_names,
        output_names=shap_values.output_names,
        output_indexes=shap_values.output_indexes,
        lower_bounds=shap_values.lower_bounds,
        upper_bounds=shap_values.upper_bounds,
        main_effects=shap_values.main_effects,
        hierarchical_values=shap_values.hierarchical_values,
        clustering=shap_values.clustering,
    )
    if return_original:
        return sv, sv_name
    else:
        return sv


def drop_description_attributes(shap_values_df, description):
    """Remove shap_values for columns that are in the description

    Args:
        shap_values_df: pd.DataFrame
        description: fairsd.Description
    """
    attributes = description.get_attributes()
    shap_values_df.drop(attributes, axis=1, errors="ignore", inplace=True)
