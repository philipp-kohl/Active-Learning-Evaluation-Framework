import mlflow
import srsly
import ale.mlflowutils.mlflow_utils as mlflow_utils

from ale.config import NLPTask
from ale.mlflowutils.ale_mlflow_artifact_files import AleArtifactFiles
from ale.pipeline.pipeline_component import PipelineComponent
from ale.pipeline.pipeline_components.utils import create_path


class CollectLabelsComponent(PipelineComponent):
    @staticmethod
    def collect_labels(
            train_path: str, dev_path: str, label_column: str, nlp_task: NLPTask
    ):
        all_labels = set()

        def cls():
            for entry in srsly.read_jsonl(train_path):
                all_labels.add(entry[label_column])

            for entry in srsly.read_jsonl(dev_path):
                all_labels.add(entry[label_column])

        def ner():
            for entry in srsly.read_jsonl(train_path):
                [all_labels.add(e[2]) for e in entry[label_column]]

            for entry in srsly.read_jsonl(dev_path):
                [all_labels.add(e[2]) for e in entry[label_column]]

        strategy = {NLPTask.CLS: cls, NLPTask.NER: ner}

        strategy[nlp_task]()
        mlflow_utils.log_dict_as_artifact(mlflow.active_run(), {"collected_labels": sorted(list(all_labels))},
                                          AleArtifactFiles.COLLECTED_LABELS.value)

    def prepare_run(self):
        data_cfg = self.pipeline_storage.cfg.data
        train_path = create_path(
            data_cfg.data_dir, data_cfg.train_file, data_cfg.file_format
        )
        dev_path = create_path(
            data_cfg.data_dir, data_cfg.dev_file, data_cfg.file_format
        )

        self.store_function(
            CollectLabelsComponent.collect_labels,
            train_path=train_path,
            dev_path=dev_path,
            label_column=data_cfg.label_column,
            nlp_task=data_cfg.nlp_task,
        )
