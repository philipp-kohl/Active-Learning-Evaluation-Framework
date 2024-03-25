from abc import ABC, abstractmethod
from typing import Dict, Mapping, List

from mlflow import ActiveRun
from mlflow.entities import Run
from torch.utils.data import DataLoader

from ale.corpus.corpus import Corpus
from ale.trainer.prediction_result import PredictionResult

# MetricsType = Dict[str, Union[str, float, "MetricsType"]]
MetricsType = Mapping[str, float]


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

    @abstractmethod
    def predict_with_known_gold_labels(self, data_loader: DataLoader) -> Dict[int, PredictionResult]:
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
