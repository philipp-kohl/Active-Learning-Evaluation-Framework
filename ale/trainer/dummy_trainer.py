import logging
from pathlib import Path
from typing import Dict

from mlflow import ActiveRun
from mlflow.entities import Run
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ale.utils import NLPTask
from ale.corpus.corpus import Corpus
from ale.mlflowutils.mlflow_utils import log_dict_as_artifact
from ale.registry.registerable_trainer import TrainerRegistry
from ale.trainer.base_trainer import MetricsType, PredictionTrainer
from ale.trainer.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


@TrainerRegistry.register("dummy-trainer")
class DummyTrainer(PredictionTrainer):
    """ """

    def __init__(
            self,
            dev_path: Path,
            test_path: Path,
            spacy_config: Path,
            use_gpu: int,
            seed: int,
            nlp_task: NLPTask,
            recreate_pipeline_each_run: bool
    ):
        logger.info("Dummy Trainer initialized!")

    def train(self, train_corpus: Corpus, active_run: ActiveRun) -> Dict[str, any]:
        logger.info("Dummy trainer train")
        metrics = {
            "cats_macro_auc": 0.5
        }
        return metrics

    def evaluate(self) -> MetricsType:
        logger.info("Evaluate on test set")
        metrics = {
            "cats_macro_auc": 0.5
        }

        return metrics

    def store_to_artifacts(self, run: Run):
        log_dict_as_artifact(run, {"key": "value"}, "model.json")
        logger.info(f"Dummy: Store model")

    def restore_from_artifacts(self, matching_run: Run):
        logger.info(f"Dummy: Restore model")

    def predict(self, docs: Dict[int, str]) -> Dict[int, PredictionResult]:
        raise NotImplemented()

    def delete_artifacts(self, run: Run):
        repository = get_artifact_repository(run.info.artifact_uri)
        repository.delete_artifacts("model.json")
