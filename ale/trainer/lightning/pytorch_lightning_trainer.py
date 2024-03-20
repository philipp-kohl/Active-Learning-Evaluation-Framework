import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from mlflow import ActiveRun
from mlflow.entities import Run
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.mlflowutils import mlflow_utils
from ale.registry import TrainerRegistry
from ale.trainer.base_trainer import BaseTrainer, MetricsType
from ale.trainer.lightning.ner_dataset import AleNerDataModule, PredictionDataModule
from ale.trainer.lightning.trf_model import TransformerLightning
from ale.trainer.prediction_result import PredictionResult

logger = logging.getLogger(__name__)


@TrainerRegistry.register("pytorch-lightning-trainer")
class PyTorchLightningTrainer(BaseTrainer):
    def __init__(self, cfg: AppConfig, corpus: Corpus, seed: int):
        self.model_class = TransformerLightning
        huggingface_model = "FacebookAI/roberta-base"
        labels = ["PER", "ORG", "LOC", "MISC"]
        seed_everything(seed, workers=True)
        self.dataset = corpus.data_module
        self.model = self.model_class(huggingface_model, labels, 2e-5, ignore_labels=["O"])
        self.cfg = cfg

    def train(self, train_corpus: Corpus, active_run: ActiveRun) -> MetricsType:
        mlf_logger = MLFlowLogger(experiment_name=self.cfg.mlflow.experiment_name,
                                  tracking_uri=self.cfg.mlflow.url,
                                  run_id=active_run.info.run_id)
        self.trainer = Trainer(max_epochs=2, devices=1, accelerator="gpu", logger=mlf_logger,
                               deterministic=True,
                               default_root_dir="pt_lightning/checkpoints/")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision('high')
        self.trainer.fit(self.model, self.dataset)
        return self.trainer.validate(ckpt_path='best', dataloaders=self.dataset.val_dataloader())[0]

    def evaluate(self) -> MetricsType:
        return self.trainer.test(ckpt_path='best', dataloaders=self.dataset.test_dataloader())[0]

    def store_to_artifacts(self, run: Run):
        model_best_path = Path(self.trainer.checkpoint_callback.best_model_path)
        renamed_path = model_best_path.with_name("model.ckpt")
        model_best_path.rename(renamed_path)
        logger.info(f"Store model to: {renamed_path}")
        mlflow_utils.log_artifact(run, renamed_path, "best")

    def restore_from_artifacts(self, matching_run: Run):
        artifact_path = "best/model.ckpt"
        logger.info(f"Restore model from: {matching_run.info.run_id}/{artifact_path}")
        model_path = mlflow_utils.load_artifact(matching_run, artifact_path)
        self.model = self.model_class.load_from_checkpoint(model_path)

    def delete_artifacts(self, run: Run):
        repository = get_artifact_repository(run.info.artifact_uri)
        repository.delete_artifacts("best")

    def predict(self, docs: Dict[int, str]) -> Dict[int, PredictionResult]:
        results: Dict[int, PredictionResult] = dict()

        texts = list(docs.values())
        data = PredictionDataModule(texts, "FacebookAI/roberta-base")
        predictions = self.trainer.predict(self.model, data.predict_dataloader())

        i = 10

        return results


