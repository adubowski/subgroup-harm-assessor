from dash import dcc, html
import plotly.express as px
from explainerdashboard.explainer_plots import plotly_confusion_matrix
from sklearn.metrics import confusion_matrix


class CMchart(html.Div):
    def __init__(self, name, y_true, y_pred):
        """
        :param name: name of the plot
        :param df: dataframe
        """
        self.html_id = name.lower().replace(" ", "-")
        self.name = name
        self.cm = confusion_matrix(y_true, y_pred)
        self.fig = plotly_confusion_matrix(self.cm)
        self.title_id = self.html_id + "-t"

        # Equivalent to `html.Div([...])`
        super().__init__()

    def update(self):
        self.fig = plotly_confusion_matrix(self.cm)

        self.fig.update_layout(
            yaxis_zeroline=False, xaxis_zeroline=False, dragmode="select"
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        # update titles
        self.fig.update_layout(
            xaxis_title=self.col1,
            yaxis_title=self.col2,
        )

        return self.fig
