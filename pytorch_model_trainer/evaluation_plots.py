# evaluation_plots.py
# %%writefile evaluation_plots.py
import os
import plotly.express as px
import pandas as pd
from typing import Tuple
from plotly.graph_objects import Figure
from plotly import io as pio

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


def nn_model_plots(saved_metrics: pd.DataFrame) -> Tuple[Figure, Figure, Figure, Figure, Figure, Figure] | Tuple[Figure, Figure, Figure]:
# Plotting curve.
    if len(saved_metrics.columns) > 7 and 'train_losses1' in saved_metrics.columns:
        fig1 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_losses0', 'val_losses0'], height=750, width=750, title='Loss Curves: GPU0')

        fig1.update_xaxes(title_text='Epoch',)
        fig1.update_yaxes(title_text='Loss')
        fig1.update_traces(mode='lines+markers')
        fig1.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig1.update_layout(legend_title_font_color="black")
        fig1.update_layout(legend_title_font_size=16)
        fig1.update_layout(legend_font_color="black")
        fig1.update_layout(legend_font_size=16)

        fig_to_html(fig1, 'loss_curves_gpu0.html')
        fig1.show()

        fig2 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_losses1', 'val_losses1'], height=750, width=750, title='Loss Curves: GPU1')

        fig2.update_xaxes(title_text='Epoch',)
        fig2.update_yaxes(title_text='Loss')
        fig2.update_traces(mode='lines+markers')
        fig2.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig2.update_layout(legend_title_font_color="black")
        fig2.update_layout(legend_title_font_size=16)
        fig2.update_layout(legend_font_color="black")
        fig2.update_layout(legend_font_size=16)

        fig_to_html(fig2, 'loss_curves_gpu1.html')
        fig2.show()

        fig3 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_accs0', 'val_accs0'], height=750, width=750, title='Accuracy Curves: GPU0')

        fig3.update_xaxes(title_text='Epoch',)
        fig3.update_yaxes(title_text='Accuracy')
        fig3.update_traces(mode='lines+markers')
        fig3.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_layout(legend_title_font_color="black")
        fig3.update_layout(legend_title_font_size=16)
        fig3.update_layout(legend_font_color="black")
        fig3.update_layout(legend_font_size=16)

        fig_to_html(fig3, 'acc_curves_gpu0.html')
        fig3.show()

        fig4 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_accs1', 'val_accs1'], height=750, width=750, title='Accuracy Curves: GPU1')

        fig4.update_xaxes(title_text='Epoch',)
        fig4.update_yaxes(title_text='Accuracy')
        fig4.update_traces(mode='lines+markers')
        fig4.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig4.update_layout(legend_title_font_color="black")
        fig4.update_layout(legend_title_font_size=16)
        fig4.update_layout(legend_font_color="black")
        fig4.update_layout(legend_font_size=16)

        fig_to_html(fig4, 'acc_curves_gpu1.html')
        fig4.show()

        fig5 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_f1s0', 'val_f1s0'], height=750, width=750, title='F1-Score Curves: GPU0')

        fig5.update_xaxes(title_text='Epoch',)
        fig5.update_yaxes(title_text='F1-Score')
        fig5.update_traces(mode='lines+markers')
        fig5.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig5.update_layout(legend_title_font_color="black")
        fig5.update_layout(legend_title_font_size=16)
        fig5.update_layout(legend_font_color="black")
        fig5.update_layout(legend_font_size=16)

        fig_to_html(fig5, 'f1s_curves_gpu0.html')
        fig5.show()

        fig6 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_f1s1', 'val_f1s1'], height=750, width=750, title='F1-Score Curves: GPU1')

        fig6.update_xaxes(title_text='Epoch',)
        fig6.update_yaxes(title_text='F1-Score')
        fig6.update_traces(mode='lines+markers')
        fig6.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig6.update_layout(legend_title_font_color="black")
        fig6.update_layout(legend_title_font_size=16)
        fig6.update_layout(legend_font_color="black")
        fig6.update_layout(legend_font_size=16)

        fig_to_html(fig6, 'f1s_curves_gpu1.html')
        fig6.show()

        return fig1, fig2, fig3, fig4, fig5, fig6
    elif len(saved_metrics.columns == 7) and 'train_losses1' not in saved_metrics.columns:
        fig1 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_losses0', 'val_losses0'], height=750, width=750, title='Loss Curves: GPU')

        fig1.update_xaxes(title_text='Epoch',)
        fig1.update_yaxes(title_text='Loss')
        fig1.update_traces(mode='lines+markers')
        fig1.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig1.update_layout(legend_title_font_color="black")
        fig1.update_layout(legend_title_font_size=16)
        fig1.update_layout(legend_font_color="black")
        fig1.update_layout(legend_font_size=16)

        fig_to_html(fig1, 'loss_curves_gpu.html')
        fig1.show()
        
        fig2 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_accs0', 'val_accs0'], height=750, width=750, title='Accuracy Curves: GPU')

        fig2.update_xaxes(title_text='Epoch',)
        fig2.update_yaxes(title_text='Accuracy')
        fig2.update_traces(mode='lines+markers')
        fig2.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig2.update_layout(legend_title_font_color="black")
        fig2.update_layout(legend_title_font_size=16)
        fig2.update_layout(legend_font_color="black")
        fig2.update_layout(legend_font_size=16)

        fig_to_html(fig2, 'acc_curves_gpu.html')
        fig2.show()
        
        fig3 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_f1s0', 'val_f1s0'], height=750, width=750, title='F1-Score Curves: GPU')

        fig3.update_xaxes(title_text='Epoch',)
        fig3.update_yaxes(title_text='F1-Score')
        fig3.update_traces(mode='lines+markers')
        fig3.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_layout(legend_title_font_color="black")
        fig3.update_layout(legend_title_font_size=16)
        fig3.update_layout(legend_font_color="black")
        fig3.update_layout(legend_font_size=16)

        fig_to_html(fig3, 'f1s_curves_gpu.html')
        fig3.show()

        return fig1, fig2, fig3
    else:
        fig1 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_losses', 'val_losses'], height=750, width=750, title='Loss Curves: CPU')

        fig1.update_xaxes(title_text='Epoch',)
        fig1.update_yaxes(title_text='Loss')
        fig1.update_traces(mode='lines+markers')
        fig1.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig1.update_layout(legend_title_font_color="black")
        fig1.update_layout(legend_title_font_size=16)
        fig1.update_layout(legend_font_color="black")
        fig1.update_layout(legend_font_size=16)

        fig_to_html(fig1, 'loss_curves_cpu.html')
        fig1.show()
        
        fig2 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_accs', 'val_accs'], height=750, width=750, title='Accuracy Curves: CPU')

        fig2.update_xaxes(title_text='Epoch',)
        fig2.update_yaxes(title_text='Accuracy')
        fig2.update_traces(mode='lines+markers')
        fig2.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig2.update_layout(legend_title_font_color="black")
        fig2.update_layout(legend_title_font_size=16)
        fig2.update_layout(legend_font_color="black")
        fig2.update_layout(legend_font_size=16)

        fig_to_html(fig2, 'acc_curves_cpu.html')
        fig2.show()
        
        fig3 = px.line(data_frame=saved_metrics, x='epochs', y=[
                    'train_f1s', 'val_f1s'], height=750, width=750, title='F1-Score Curves: CPU')

        fig3.update_xaxes(title_text='Epoch',)
        fig3.update_yaxes(title_text='F1-Score')
        fig3.update_traces(mode='lines+markers')
        fig3.update_layout(legend_title_text='Metrics:', legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_layout(legend_title_font_color="black")
        fig3.update_layout(legend_title_font_size=16)
        fig3.update_layout(legend_font_color="black")
        fig3.update_layout(legend_font_size=16)

        fig_to_html(fig3, 'f1s_curves_cpu.html')
        fig3.show()

        return fig1, fig2, fig3
