import os

from mlflow.artifacts import download_artifacts

from ale.data.data import load_local_data
from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.components import PipelineComponents


class LoadDataConvertedComponent(PipelineComponent):
    def prepare_run(self):
        convert_run = self.pipeline_storage.completed_runs[
            PipelineComponents.CONVERT_DATA
        ]
        dir = os.path.join(download_artifacts(artifact_uri=convert_run.info.artifact_uri), "data")

        data_cfg = self.pipeline_storage.cfg.data
        self.store_function(
            load_local_data,
            data_dir=dir,
            train_file=data_cfg.train_file,
            test_file=data_cfg.test_file,
            dev_file=data_cfg.dev_file,
            file_format=self.pipeline_storage.cfg.converter.target_format,
        )
