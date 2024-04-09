import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from mlflow import ActiveRun
from mlflow.entities import Run
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from torch.utils.data import DataLoader

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.mlflowutils import mlflow_utils
from ale.registry import TrainerRegistry
from ale.registry.registerable_model import ModelRegistry
from ale.trainer.base_trainer import BaseTrainer, MetricsType
from ale.trainer.lightning.ner_dataset import PredictionDataModule
from ale.trainer.prediction_result import PredictionResult, TokenConfidence, LabelConfidence

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('medium')

logger = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)


@TrainerRegistry.register("pytorch-lightning-trainer")
class PyTorchLightningTrainer(BaseTrainer):
    def __init__(self, cfg: AppConfig, corpus: Corpus, seed: int, labels: List[str]):
        seed_everything(seed, workers=True)
        self.dataset = corpus.data_module
        self.model_class = ModelRegistry.get_instance(cfg.trainer.model)
        self.model = self.model_class(cfg.trainer.huggingface_model,
                                      labels,
                                      cfg.trainer.learning_rate,
                                      cfg.trainer.weight_decay,
                                      ignore_labels=["O"],
                                      label_smoothing=cfg.trainer.label_smoothing)
        self.cfg = cfg

    def train(self, train_corpus: Corpus, active_run: ActiveRun) -> MetricsType:
        self.create_trainer(active_run)
        self.trainer.fit(self.model, self.dataset)
        return self.trainer.validate(ckpt_path='best', dataloaders=self.dataset.val_dataloader())[0]

    def create_trainer(self, active_run):
        mlf_logger = MLFlowLogger(experiment_name=self.cfg.mlflow.experiment_name,
                                  tracking_uri=self.cfg.mlflow.url,
                                  run_id=active_run.info.run_id)
        early_stop_callback = EarlyStopping(monitor="val_f1_macro",
                                            min_delta=self.cfg.trainer.early_stopping_delta,
                                            patience=self.cfg.trainer.early_stopping_patience,
                                            verbose=True, mode="max")
        checkpoint_callback = ModelCheckpoint(dirpath='pt_lightning/checkpoints/', save_top_k=2, save_weights_only=True,
                                              save_on_train_epoch_end=True, monitor="val_f1_macro", mode="max")
        callbacks = [early_stop_callback, checkpoint_callback]
        self.trainer = Trainer(max_epochs=self.cfg.trainer.max_epochs, devices=1, accelerator=self.cfg.trainer.device,
                               logger=mlf_logger, deterministic=True,
                               # profiler="simple"
                               callbacks=callbacks
                               )

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
        data = PredictionDataModule(texts,
                                    self.cfg.trainer.huggingface_model,
                                    num_workers=self.cfg.trainer.num_workers)
        if hasattr(self, 'trainer'):
            logging.info("Reuse trainer from training for predictions")
            trainer = self.trainer
        else:
            logger.warning("No trainer is defined. Might be after resume. Let's try a fresh trainer.")
            trainer = Trainer(max_epochs=self.cfg.trainer.max_epochs, devices=1, accelerator=self.cfg.trainer.device,
                              deterministic=True)

        prediction_batches = trainer.predict(self.model, data.predict_dataloader())
        predictions_per_doc = []
        for single_batch in prediction_batches:
            first_key = list(single_batch.keys())[0]
            for i in range(len(single_batch[first_key])):
                infos_per_doc = {}
                for key in single_batch:
                    infos_per_doc[key] = single_batch[key][i]
                predictions_per_doc.append(infos_per_doc)

        for idx, pred in zip(docs.keys(), predictions_per_doc):
            prediction_result = PredictionResult()
            for single_token, single_conf_array in zip(pred['tokens'], pred['confidences']):
                label_confidences = [LabelConfidence(label=l, confidence=c) for l, c in single_conf_array.items()]
                prediction_result.ner_confidences_token.append(
                    TokenConfidence(text=single_token, label_confidence=label_confidences))

            results[idx] = prediction_result

        return results

    def predict_with_known_gold_labels(self, data_loader: DataLoader) -> Dict[int, PredictionResult]:
        keys = [entry["id"] for entry in data_loader.dataset]
        prediction_batches = self.trainer.predict(self.model, data_loader)

        results: Dict[int, PredictionResult] = dict()
        predictions_per_doc = []

        for single_batch in prediction_batches:
            first_key = list(single_batch.keys())[0]
            for i in range(len(single_batch[first_key])):
                infos_per_doc = {}
                for key in single_batch:
                    infos_per_doc[key] = single_batch[key][i]
                predictions_per_doc.append(infos_per_doc)

        for idx, pred in zip(keys, predictions_per_doc):
            prediction_result = PredictionResult()
            for single_token, single_conf_array, gold_label in zip(pred['tokens'], pred['confidences'],
                                                                   pred['gold_labels']):
                label_confidences = [LabelConfidence(label=l, confidence=c) for l, c in single_conf_array.items()]
                prediction_result.ner_confidences_token.append(
                    TokenConfidence(text=single_token, label_confidence=label_confidences,
                                    gold_label=gold_label))

            results[idx] = prediction_result

        return results
