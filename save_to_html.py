# Function for saving plotly plots as html to embed them later
import os
import plotly

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
    fig: plotly.graph_objs.Figure,
    # plot_heading: str,
    output_path: str = "output.html",
    template_path: str= "html_template.html",
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
