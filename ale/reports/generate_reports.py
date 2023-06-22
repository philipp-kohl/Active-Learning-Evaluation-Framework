from collections import defaultdict
from typing import List, Optional

import click
import numpy as np
from mlflow import MlflowClient
import plotly.graph_objs as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None


line_style = ["dash", "dot", "dashdot"]

@click.command()
@click.option(
    "--runs",
    type=str,
    required=True,
    multiple=True,
    help="Run ids for different strategies to compare",
)
@click.option(
    "--names",
    type=str,
    required=False,
    multiple=True,
    help="Display name for each run name",
)
@click.option(
    "--tracking-url",
    type=str,
    required=False,
    help="Mlflow tracking url. Defaults to localhost with port 5000",
    default="http://localhost:5000",
)
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
def main(runs: List[str], tracking_url: str, out: Optional[str], y_axis: Optional[str], names: Optional[List[str]]):
    client = MlflowClient(tracking_uri=tracking_url)
    mlflow_runs = [client.get_run(run_id) for run_id in runs]

    scatter_plots = []
    for idx, run in enumerate(mlflow_runs):
        experiment = client.get_experiment(run.info.experiment_id)
        metrics = run.data.metrics

        metric_dict = extract_means_and_deviations_per_metric(client, metrics, run)
        for metric in metric_dict:
            name = f"{experiment.name}-{run.info.run_name}-{metric}"
            name = f"{experiment.name}"

            current_line_style = line_style[len(line_style) % (idx +1 )]
            scatters = create_line_with_error_bars(name, metric, metric_dict, current_line_style)

            scatter_plots.extend(scatters)

    fig = go.Figure(scatter_plots, layout=go.Layout(
        xaxis={"title": "Number of Datapoints"},
        yaxis={"title": y_axis},
    ))
    fig.show()
    fig.write_image(out)


def extract_means_and_deviations_per_metric(client, metrics, run):
    metric_dict = defaultdict(dict)
    for metric in metrics:
        metrics_history = client.get_metric_history(run.info.run_id, metric)

        if "-deviations" in metric:
            metric_name = metric.split("-deviations")[0]
            metric_dict[metric_name]["deviations"] = [m.value for m in metrics_history]
            metric_dict[metric_name]["steps"] = [m.step for m in metrics_history]
        elif "-means" in metric:
            metric_name = metric.split("-means")[0]
            metric_dict[metric_name]["means"] = [m.value for m in metrics_history]
        else:
            raise ValueError(
                f"{metric} does not follow (-means, -deviations) conventions!"
            )

    # TODO remove after investigation and fix
    means = []
    devs = []
    steps = []
    for mean, dev, step in zip(metric_dict[metric_name]["means"], metric_dict[metric_name]["deviations"], metric_dict[metric_name]["steps"]):
        if not np.isnan(dev):
            means.append(mean)
            devs.append(dev)
            steps.append(step)
    metric_dict[metric_name]["means"] = means
    metric_dict[metric_name]["deviations"] = devs
    metric_dict[metric_name]["steps"] = steps
    return metric_dict


def create_line_with_error_bars(name, metric, metric_dict, line_type):
    return [
        go.Scatter(
            name=name,
            x=metric_dict[metric]["steps"],
            y=metric_dict[metric]["means"],
            mode="lines",
            legendgroup=name,
            #line={"dash": line_type}
        ),
        go.Scatter(
            name="Upper Bound",
            x=metric_dict[metric]["steps"],
            y=[
                mean + dev
                for mean, dev in zip(
                    metric_dict[metric]["means"], metric_dict[metric]["deviations"]
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
            x=metric_dict[metric]["steps"],
            y=[
                mean - dev
                for mean, dev in zip(
                    metric_dict[metric]["means"], metric_dict[metric]["deviations"]
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
