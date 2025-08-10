# Plotting ROC Curve
import plotly
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import spmatrix
from save_to_html import fig_to_html
from typing import Union
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.io as pio

pio.renderers.default = "colab"
pio.templates.default = "seaborn"

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> plotly.graph_objs.Figure:
    """Plots the ROC curve."""

    # 1. Binarize the labels.
    n_classes = len(target_names)  # Get the number of classes
    y_true_bin: Union[np.ndarray, spmatrix] = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin: Union[np.ndarray, spmatrix] = label_binarize(y_pred, classes=range(n_classes))

    # Convert to dense arrays if they are sparse matrices
    y_true_bin = np.asarray(y_true_bin)
    y_pred_bin = np.asarray(y_pred_bin)

    # 2. Create the figure.
    fig = go.Figure()

    # 3. Calculate the fpr and tpr.
    for i, k in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{k} (AUC = {roc_auc:.2f})")
        )

    # 4. Update the plot.
    fig.update_layout(
        title=f"ROC Curve: {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        showlegend=True,
    )

    fig_to_html(fig, f"{plot_name}")

    fig.show()  # Display
    return fig
