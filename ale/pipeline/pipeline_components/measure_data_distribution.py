from pathlib import Path

import mlflow
import srsly
import ale.mlflowutils.mlflow_utils as mlflow_utils
from ale.bias.data_distribution import DataDistribution

from ale.config import NLPTask
from ale.mlflowutils.ale_mlflow_artifact_files import AleArtifactFiles
from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.pipeline_components.utils import create_path


class DataDistributionMeasure(PipelineComponent):
    @staticmethod
    def collect_labels(
            train_path: str, dev_path: str, test_path: str, label_column: str, nlp_task: NLPTask
    ):
        DataDistributionMeasure.log_data_distribution(label_column, nlp_task, train_path, "train/data_distribution")
        DataDistributionMeasure.log_data_distribution(label_column, nlp_task, dev_path, "dev/data_distribution")
        DataDistributionMeasure.log_data_distribution(label_column, nlp_task, test_path, "test/data_distribution")

    @staticmethod
    def log_data_distribution(label_column, nlp_task, train_path, artifact_name: str):
        distribution = DataDistribution(nlp_task, label_column, Path(train_path))
        dist_by_label = distribution.get_data_distribution_by_label()
        distribution.store_distribution(dist_by_label, mlflow.active_run(), artifact_name)

    def prepare_run(self):
        data_cfg = self.pipeline_storage.cfg.data
        train_path = create_path(
            data_cfg.data_dir, data_cfg.train_file, data_cfg.file_format
        )
        dev_path = create_path(
            data_cfg.data_dir, data_cfg.dev_file, data_cfg.file_format
        )
        test_path = create_path(
            data_cfg.data_dir, data_cfg.test_file, data_cfg.file_format
        )

        self.store_function(
            DataDistributionMeasure.collect_labels,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            label_column=data_cfg.label_column,
            nlp_task=data_cfg.nlp_task,
        )
