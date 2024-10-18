import logging
from collections import defaultdict
from typing import List

import mlflow
import pandas as pd
import plotly.graph_objs as go
from mlflow.entities import RunStatus, ViewType, Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from pandas import DataFrame

from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_component import PipelineComponent

logger = logging.getLogger(__name__)


class AggregateSeedRuns(PipelineComponent):
    def plot_metric(self, means: DataFrame, deviations: DataFrame, name: str):
        means["x"] = means.index
        means["upper"] = means["value"] + deviations["value"]
        means["lower"] = means["value"] - deviations["value"]
        fig = go.Figure(
            [
                go.Scatter(
                    name="Measurement",
                    x=means["x"],
                    y=means["value"],
                    mode="lines",
                    line=dict(color="rgb(31, 119, 180)"),
                ),
                go.Scatter(
                    name="Upper Bound",
                    x=means["x"],
                    y=means["upper"],
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Lower Bound",
                    x=means["x"],
                    y=means["lower"],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        fig.update_layout(
            title=name,
        )

        return fig

    def agg(self):
        propose_data_run = self.pipeline_storage.completed_runs[
            PipelineComponents.SEED_RUNS
        ]
        runs = self.get_child_runs(propose_data_run.info.run_id)

        client = mlflow.tracking.MlflowClient()
        metrics = self.pipeline_storage.cfg.experiment.tracking_metrics

        tmp = defaultdict(list)

        parent_run_id = mlflow.active_run().data.tags.get(MLFLOW_PARENT_RUN_ID)

        for metric in metrics:
            for run in runs:
                metric_values = client.get_metric_history(run.info.run_id, metric)
                for m in metric_values:
                    tmp["metric"].append(metric)
                    tmp["step"].append(m.step)
                    tmp["value"].append(m.value)

        df = pd.DataFrame(data=tmp)
        means = df.groupby(["metric", "step"]).mean()
        deviations = df.groupby(["metric", "step"]).std()

        for metric in metrics:
            fig = self.plot_metric(means.loc[metric], deviations.loc[metric], metric)
            mlflow.log_figure(fig, f"{metric}.html")

            for step, value in means.loc[metric]["value"].items():
                client.log_metric(
                    parent_run_id, key=f"{metric}-means", value=value, step=step
                )

            for step, value in deviations.loc[metric]["value"].items():
                client.log_metric(
                    parent_run_id, key=f"{metric}-deviations", value=value, step=step
                )

    def prepare_run(self):
        self.store_function(self.agg)

    def get_child_runs(self, parent_run_id: str) -> List[Run]:
        client = mlflow.tracking.MlflowClient()
        filter_string = (
            f"attributes.status = '{RunStatus.to_string(RunStatus.FINISHED)}' "
            f"and tags.mlflow.parentRunId = '{parent_run_id}' "
        )
        all_run_infos = client.search_runs(
            experiment_ids=[self.pipeline_storage.experiment_id],
            filter_string=filter_string,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.end_time DESC"],
        )

        return [run for run in all_run_infos]
