import os
from pathlib import Path
from typing import List, Any

import srsly
from mlflow.artifacts import download_artifacts

from ale.config import AppConfig
from ale.mlflowutils.ale_mlflow_artifact_files import AleArtifactFiles
from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_component import PipelineComponent
from ale.proposer.proposer import AleBartender


class ProposeDataComponent(PipelineComponent):
    @staticmethod
    def propose_data(data_converted: Path, data_raw: Path, cfg: AppConfig, labels: List[Any]):
        proposer = AleBartender(data_converted, data_raw, cfg, labels)
        proposer.run()

    def prepare_run(self):
        load_collected_labels_run = self.pipeline_storage.completed_runs[
            PipelineComponents.COLLECT_LABELS
        ]
        downloaded_path = download_artifacts(run_id=load_collected_labels_run.info.run_id,
                                             artifact_path=AleArtifactFiles.COLLECTED_LABELS.value)
        json = srsly.read_json(downloaded_path)
        labels = json["collected_labels"]

        load_data_run_raw_run = self.pipeline_storage.completed_runs[
            PipelineComponents.LOAD_DATA_RUN_RAW
        ]
        load_data_run_converted_run = self.pipeline_storage.completed_runs[
            PipelineComponents.LOAD_DATA_RUN_CONVERTED
        ]

        data_raw_uri = os.path.join(
            download_artifacts(artifact_uri=load_data_run_raw_run.info.artifact_uri), "data"
        )
        data_converted_uri = os.path.join(
            download_artifacts(artifact_uri=load_data_run_converted_run.info.artifact_uri), "data"
        )

        self.store_function(
            ProposeDataComponent.propose_data,
            data_converted=Path(data_converted_uri),
            data_raw=Path(data_raw_uri),
            cfg=self.pipeline_storage.cfg,
            labels=labels
        )
