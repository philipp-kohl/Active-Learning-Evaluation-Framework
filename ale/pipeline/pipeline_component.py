import logging
from abc import ABC, abstractmethod

import mlflow
from mlflow import ActiveRun
from mlflow.entities import RunStatus

from ale.mlflowutils.mlflow_utils import _already_ran, walk_params_from_omegaconf_dict, store_log_file_to_mlflow
from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


class PipelineComponent(ABC):
    def __init__(self, run_name: PipelineComponents, pipeline_storage: PipelineStorage):
        self.run_name = run_name
        self.pipeline_storage = pipeline_storage
        self.parameters = None
        self.function = None

    def run(self):
        logger.info(f"Start pipeline component {self.run_name}")
        self.prepare_run()

        if self.parameters is None or self.function is None:
            raise ValueError(
                "Parameters and function must be set via store_function(...) in the prepare_run() method!"
            )

        matching_run = _already_ran(
            self.parameters,
            self.pipeline_storage.git_commit,
            experiment_id=self.pipeline_storage.experiment_id,
            run_name=self.run_name.value,
            run_status=RunStatus.FINISHED,
        )
        if matching_run is not None:
            self.pipeline_storage.completed_runs[self.run_name] = matching_run
            return matching_run

        matching_run_failed_before = _already_ran(
            self.parameters,
            self.pipeline_storage.git_commit,
            experiment_id=self.pipeline_storage.experiment_id,
            run_name=self.run_name.value,
            run_status=RunStatus.FAILED,
        )

        run_id = None
        # Try to resume run if failed run was found
        if matching_run_failed_before:
            logger.warning(f"Try resuming failed run({matching_run_failed_before.info.run_id}) and resume it!")
            run_id = matching_run_failed_before.info.run_id

        matching_running_before = _already_ran(
            self.parameters,
            self.pipeline_storage.git_commit,
            experiment_id=self.pipeline_storage.experiment_id,
            run_name=self.run_name.value,
            run_status=RunStatus.RUNNING,
        )
        # Try to resume run if run was not marked as failed and is still marked as running...
        if matching_running_before:
            logger.warning(f"Try resuming running run({matching_running_before.info.run_id}) and resume it!")
            run_id = matching_running_before.info.run_id

        with mlflow.start_run(
            experiment_id=self.pipeline_storage.experiment_id,
            nested=True,
            run_name=self.run_name.value,
            run_id=run_id,
        ) as active_mlflow_run:
            walk_params_from_omegaconf_dict(
                self.parameters, lambda name, value: mlflow.log_param(name, value)
            )
            self.function(**self.parameters)

        self.pipeline_storage.completed_runs[self.run_name] = mlflow.get_run(
            active_mlflow_run.info.run_id
        )
        self.after_call(active_mlflow_run)

    def store_function(self, function, **kwargs):
        self.parameters = kwargs
        self.function = function

    def after_call(self, active_mlflow_run: ActiveRun):
        """
        Method called after run has finished. Can be used to store values in the pipeline_storage.
        """
        store_log_file_to_mlflow("main.log", active_mlflow_run.info.run_id)
        pass

    @abstractmethod
    def prepare_run(self):
        pass
