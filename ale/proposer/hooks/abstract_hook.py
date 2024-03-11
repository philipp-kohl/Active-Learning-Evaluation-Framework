from typing import List, Optional, Dict

from mlflow import ActiveRun
from mlflow.entities import Run

from ale.config import AppConfig
from ale.corpus.corpus import Corpus
from ale.trainer.prediction_result import PredictionResult


class ProposeHook:
    def __init__(self, cfg: AppConfig, parent_run_id: str, corpus: Corpus, **kwargs):
        self.cfg = cfg
        self.parent_run_id = parent_run_id
        self.corpus = corpus
        self.kwargs = kwargs

    def before_proposing(self) -> None:
        pass

    def after_proposing(self) -> None:
        pass

    def after_training(self, mlflow_run: Run) -> None:
        pass

    def on_iter_end(self) -> None:
        pass

    def on_seed_end(self) -> None:
        pass

    def after_prediction(self,
                         mlflow_run: Run,
                         preds_train: Optional[Dict[int, PredictionResult]],
                         preds_dev: Optional[Dict[int, PredictionResult]]) -> None:
        pass

    def needs_dev_predictions(self) -> bool:
        return False

    def needs_train_predictions(self) -> bool:
        return False
