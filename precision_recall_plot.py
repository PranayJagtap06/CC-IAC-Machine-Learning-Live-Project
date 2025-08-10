# Plotting Precision-Recall Curve
import plotly
import numpy as np
import plotly.graph_objects as go
from scipy.sparse import spmatrix
from save_to_html import fig_to_html
from typing import Union
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import plotly.io as pio

pio.renderers.default = "colab"
pio.templates.default = "seaborn"

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> plotly.graph_objs.Figure:
    """Plot precision-recall curve."""

    # Assuming you have 'y_test' (true labels) and 'y_preds' (predicted labels)

    # 1. Binarize the labels
    n_classes = len(target_names)  # Get the number of classes
    y_true_bin: Union[np.ndarray, spmatrix] = label_binarize(y_true, classes=range(n_classes))
    y_pred_bin: Union[np.ndarray, spmatrix] = label_binarize(y_pred, classes=range(n_classes))

    # Convert to dense arrays if they are sparse matrices
    y_true_bin = np.asarray(y_true_bin)
    y_pred_bin = np.asarray(y_pred_bin)

    # 2. Create the Plotly figure
    fig = go.Figure()

    # 3. Calculate and plot precision-recall curves for each class
    for i, k in enumerate(target_names):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], y_pred_bin[:, i]
        )
        avg_precision = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{k} (Avg Precision: {avg_precision:.2f})",
            )
        )

    # 4. Update layout for better visualization
    fig.update_layout(
        title=f"Precision-Recall Curve: {model_name}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        showlegend=True,
    )

    fig_to_html(fig, f"{plot_name}")

    fig.show()  # Display plot
    return fig
