import logging
import signal
import sys
from typing import List

import mlflow
from haikunator import Haikunator
from mlflow import MlflowClient
from mlflow.entities import RunStatus, ViewType
from mlflow.utils import mlflow_tags
from omegaconf import OmegaConf

from ale.config import AppConfig
from ale.mlflowutils.mlflow_utils import (
    get_git_revision_hash,
    _already_ran,
    walk_params_from_omegaconf_dict,
)
from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


class MLFlowPipeline:
    def __init__(self, cfg: AppConfig):
        self.pipeline_storage = PipelineStorage()
        self.pipeline_components: List[PipelineComponent] = []
        self.pipeline_storage.cfg = cfg

    def add(self, run_name: PipelineComponents, pipeline_component_class):
        self.pipeline_components.append(
            pipeline_component_class(run_name, self.pipeline_storage)
        )

    def get_experiment_id(self) -> str:
        """ """
        cfg = self.pipeline_storage.cfg
        if OmegaConf.is_missing(cfg.mlflow, "experiment_name"):
            experiment_name = Haikunator().haikunate()
        else:
            experiment_name = cfg.mlflow.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return mlflow.create_experiment(experiment_name)
        else:
            return experiment.experiment_id

    def start(self):
        logger.info(
            f"Starting pipeline: {[component.run_name.value for component in self.pipeline_components]}"
        )
        experiment_id = self.get_experiment_id()
        self.pipeline_storage.experiment_id = experiment_id
        cfg = self.pipeline_storage.cfg
        run_name = cfg.mlflow.run_name

        if run_name is None:
            run_name = f"{cfg.teacher.strategy}-{Haikunator().haikunate()}"
            logger.info(f"No run name specified. Use generated: {run_name}")

        def handler_stop_signals(_signo, _stack_frame):
            logger.error("Process killed. Try to exit gracefully and mark manual opened runs as failed.")
            filter_string = f"attributes.status = '{RunStatus.to_string(RunStatus.RUNNING)}'"
            # if run_name:
            #     filter_string += f" and tags.mlflow.runName = '{run_name}'"

            client = MlflowClient()
            all_run_infos = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=filter_string,
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=["attributes.end_time DESC"],
            )

            for run in all_run_infos:
                logger.error(f"Mark run ({run.info.run_id}) as failed.")
                client.set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.FAILED))

            sys.exit(0)

        signal.signal(signal.SIGTERM, handler_stop_signals)
        signal.signal(signal.SIGINT, handler_stop_signals)

        matching_run = _already_ran(
            cfg,
            get_git_revision_hash(),
            run_name=run_name,
            experiment_id=self.pipeline_storage.experiment_id,
            run_status=RunStatus.FAILED,
        )

        if matching_run is None:
            matching_run = _already_ran(
                cfg,
                get_git_revision_hash(),
                run_name=run_name,
                experiment_id=self.pipeline_storage.experiment_id,
                run_status=RunStatus.RUNNING # TODO is that okay?
            )

        with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
            run_id=matching_run.info.run_id if matching_run else None,
        ) as active_run:
            walk_params_from_omegaconf_dict(
                cfg, lambda name, value: mlflow.log_param(name, value)
            )
            self.pipeline_storage.git_commit = active_run.data.tags.get(
                mlflow_tags.MLFLOW_GIT_COMMIT
            )

            for pipeline_component in self.pipeline_components:
                pipeline_component.run()
