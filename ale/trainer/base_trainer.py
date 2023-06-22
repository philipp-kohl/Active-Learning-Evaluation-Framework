from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from mlflow import ActiveRun
from mlflow.entities import Run

from ale.corpus.corpus import Corpus
from ale.trainer.prediction_result import PredictionResult

MetricsType = Dict[str, Union[str, float, "MetricsType"]]


class BaseTrainer(ABC):
    """
    Base class for all trainers
    """

    @abstractmethod
    def train(self, train_corpus: Corpus, active_run: ActiveRun) -> MetricsType:
        pass

    @abstractmethod
    def evaluate(self) -> MetricsType:
        pass

    @abstractmethod
    def store_to_artifacts(self, run: Run):
        pass

    @abstractmethod
    def restore_from_artifacts(self, matching_run: Run):
        pass

    @abstractmethod
    def delete_artifacts(self, run: Run):
        pass

class Predictor(ABC):
    """
    Mixin for all prediction trainers to implement the predict method.
    """

    @abstractmethod
    def predict(self, docs: Dict[int, str]) -> Dict[int, PredictionResult]:
        ...


class PredictionTrainer(BaseTrainer, Predictor):
    """
    Trainer for prediction tasks.
    """
