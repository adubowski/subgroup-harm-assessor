import pandas as pd
from metrics import get_name_from_metric_str
import plotly.express as px


def get_subgroup_dropdown_options(result_set_df: pd.DataFrame, quality_metric: str = "Quality"):
    """Get the options for the subgroup dropdown. They consist of the subgroup description and the subgroup index."""

    if result_set_df.empty:
        return [{"label": "No subgroups found. Check your subgroup search criteria.", "value": -1}]
        
    # For each description, get the size and quality of the corresponding subgroup
    return [
        {
            "label": str(result_set_df["description"][idx])
                        + "; Size: "
                        + str(result_set_df["size"][idx])
                        # + f"; {quality_metric}: "
                        # + str(result_set_df["quality"][idx].round(3))
                        # Add the metric value (e.g. Accuracy for acc_diff)
                        + f"; {get_name_from_metric_str(quality_metric)}: "
                        + str(result_set_df['metric_score'][idx]),
            "value": idx,
        }
        for idx in range(len(result_set_df))
    ]


def get_shap_barchart(shap_values_df, group_feature='group', title="Feature contributions"):
        # Sort the values by mean
        agg_df = shap_values_df.groupby(group_feature).mean()
        print(agg_df)
        # Create the fig
        fig = px.bar(
            agg_df,
            y=agg_df.index,
            x=agg_df.values,
            color=agg_df.index,
            barmode="group",
            title=title,
            orientation="h",
        )

        # Update the fig
        fig.update_layout(
            title=title,
            xaxis_title="Contribution (logloss SHAP)",
            yaxis_title="",
        )
        # Turn y labels
        fig.update_layout(yaxis_tickangle=-35)
        # Set x axis range 
        fig.update_xaxes(range=[-0.1, 0.1]) 
        return fig