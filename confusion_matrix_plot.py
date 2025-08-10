# Plotting the Confusion Matrix
import plotly
import numpy as np
import plotly.express as px
from scipy.sparse import spmatrix
from save_to_html import fig_to_html
from typing import Union, Tuple
from sklearn.metrics import confusion_matrix
import plotly.io as pio

pio.renderers.default = "colab"
pio.templates.default = "seaborn"

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> plotly.graph_objs.Figure:
    """Plot confusion matrix."""

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,  # Display values on the heatmap
        labels=dict(x="Predicted", y="True"),  # Set axis labels
        x=target_names,  # Update x-axis labels
        y=target_names,  # Update y-axis labels
        color_continuous_scale="Blues",  # Customize the color scale
        width=1000,
        height=1000,
    )

    fig.update_layout(title=f"Confusion Matrix: {model_name}")  # Set plot title
    fig_to_html(fig, f"{plot_name}")
    fig.show()  # Display plot
    return fig
