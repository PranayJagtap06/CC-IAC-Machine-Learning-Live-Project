# evaluation_plots.py
import os
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, Tuple
from scipy.sparse import spmatrix
from plotly.graph_objs import Figure
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)

pio.renderers.default = "browser"  # 'colab' for colab env
pio.templates.default = "seaborn"

# Function for saving plotly plots as html to embed them later
with open("html_template.html", "w") as f:
    f.write("""
  <!doctype html>
  <html>
  <head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  </head>

  <body>
  <!-- <h3>{{ heading }}</h3> -->
  {{ fig }}
  </body>
  </head>
  """)


def fig_to_html(
    fig: Figure,
    # plot_heading: str,
    output_path: str = "output.html",
    template_path: str = "html_template.html",
) -> None:
    """
    Convert a plotly figure to an HTML.
    """
    # Create output directory if it doesn't exist
    output_dir = "plotly_html"
    os.makedirs(output_dir, exist_ok=True)

    from jinja2 import Template

    # Convert the figure to HTML
    plotly_jinja_data = {
        "fig": fig.to_html(full_html=False, include_plotlyjs="cdn"),
        # "heading": plot_heading
    }

    # Load the template
    with open(os.path.join(output_dir, output_path), "w", encoding="utf-8") as f:
        with open(template_path, "r", encoding="utf-8") as template_file:
            template = Template(template_file.read())
            f.write(template.render(plotly_jinja_data))


# Plotting the Confusion Matrix
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> Figure:
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

    # Set plot title
    fig.update_layout(title=f"Confusion Matrix: {model_name}")
    fig_to_html(fig, f"{plot_name}")
    fig.show()  # Display plot
    return fig  # Return the figure object


# Plotting Precision-Recall Curve
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> Figure:
    """Plot precision-recall curve."""

    # Assuming you have 'y_test' (true labels) and 'y_preds' (predicted labels)

    # 1. Binarize the labels
    n_classes = len(target_names)  # Get the number of classes
    y_true_bin: Union[np.ndarray, spmatrix] = label_binarize(
        y_true, classes=range(n_classes))
    y_pred_bin: Union[np.ndarray, spmatrix] = label_binarize(
        y_pred, classes=range(n_classes))

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
        avg_precision = average_precision_score(
            y_true_bin[:, i], y_pred_bin[:, i])

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
    return fig  # Return the figure object


# Plotting ROC Curve


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: np.ndarray,
    model_name: str,
    plot_name: str,
) -> Figure:
    """Plots the ROC curve."""

    # 1. Binarize the labels.
    n_classes = len(target_names)  # Get the number of classes
    y_true_bin: Union[np.ndarray, spmatrix] = label_binarize(
        y_true, classes=range(n_classes))
    y_pred_bin: Union[np.ndarray, spmatrix] = label_binarize(
        y_pred, classes=range(n_classes))

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
            go.Scatter(x=fpr, y=tpr, mode="lines",
                       name=f"{k} (AUC = {roc_auc:.2f})")
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
    return fig  # Return the figure object
