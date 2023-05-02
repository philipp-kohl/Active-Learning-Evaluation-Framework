from typing import List

import click
import pandas as pd
import plotly.graph_objs as go


line_style = ["dash", "dot", "dashdot"]

INPUT_PATH = "output_csvs/trec_coarse/"
@click.command()
@click.option(
    "--out",
    type=str,
    required=False,
    help="Path of the output file (pdf, png, jpeg, svg, eps)",
    default=None,
)
@click.option(
    "--y-axis",
    type=str,
    required=False,
    help="Name of the y-axis",
    default=None,
)
@click.option(
    "--runs",
    type=str,
    required=True,
    multiple=True,
    help="Run ids for different strategies to compare",
)
def main(runs: List[str], out: str, y_axis: str):
    scatter_plots = []
    for idx, run in enumerate(runs):
        input_path = INPUT_PATH + run
        means_csv = input_path + "_means.csv"
        deviations_csv = input_path + "_deviations.csv"
        df = pd.read_csv(means_csv)
        df.rename(columns={"value": "mean"}, inplace=True)
        df["deviation"] = pd.read_csv(deviations_csv)["value"]
        current_line_style = line_style[len(line_style) % (idx + 1)]
        scatters = create_line_with_error_bars(name=run, df=df, line_type=current_line_style)

        scatter_plots.extend(scatters)

    fig = go.Figure(scatter_plots, layout=go.Layout(
        xaxis={"title": "Number of Datapoints"},
        yaxis={"title": y_axis},
    ))
    fig.show()
    fig.write_image(out)


def create_line_with_error_bars(name, df, line_type):
    return [
        go.Scatter(
            name=name,
            x=df["step"],
            y=df["mean"],
            mode="lines+markers",
            legendgroup=name,
            #line={"dash": line_type}
        ),
        go.Scatter(
            name="Upper Bound",
            x=df.step,
            y=[
                mean + dev
                for mean, dev in zip(
                    df["mean"], df["deviation"]
                )
            ],
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False,
            legendgroup=name,
        ),
        go.Scatter(
            name="Lower Bound",
            x=df.step,
            y=[
                mean - dev
                for mean, dev in zip(
                    df["mean"], df["deviation"]
                )
            ],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines",
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
            showlegend=False,
            legendgroup=name,
        ),
    ]


if __name__ == "__main__":
    main()
