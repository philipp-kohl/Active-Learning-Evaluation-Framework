import ast
import logging

import mlflow
import srsly

from ale.mlflowutils.ale_mlflow_artifact_files import AleArtifactFiles
from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.components import PipelineComponents
from ale.pipeline.pipeline_components.utils import prepare_data
from ale.preprocessing.spacy.convert_jsonl_to_spacy_bin import (
    convert_json_to_spacy_doc_bin,
)
from ale.registry.registerable_pipeline_component import PipelineComponentRegistry

logger = logging.getLogger(__name__)


@PipelineComponentRegistry.register("jsonl_to_docbin")
class ConvertJsonlToSpacyComponent(PipelineComponent):
    def prepare_run(self):
        train_path, dev_path, test_path = prepare_data(self.pipeline_storage.cfg)
        collect_labels_run = self.pipeline_storage.completed_runs[
            PipelineComponents.COLLECT_LABELS
        ]

        downloaded_path = mlflow.artifacts.download_artifacts(run_id=collect_labels_run.info.run_id,
                                                              artifact_path=AleArtifactFiles.COLLECTED_LABELS.value)
        json = srsly.read_json(downloaded_path)
        all_labels = json["collected_labels"]

        self.store_function(
            convert_json_to_spacy_doc_bin,
            input_path=[train_path, dev_path, test_path],
            text_column=self.pipeline_storage.cfg.data.text_column,
            label_column=self.pipeline_storage.cfg.data.label_column,
            id_column="id",
            label=all_labels,
            force=False,
            language=self.pipeline_storage.cfg.trainer.language,
            nlp_task=self.pipeline_storage.cfg.data.nlp_task,
        )
