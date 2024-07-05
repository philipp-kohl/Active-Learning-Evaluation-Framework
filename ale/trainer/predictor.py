from abc import ABC, abstractmethod
from typing import Dict

from ale.trainer.prediction_result import PredictionResult


class Predictor(ABC):
    """
    Mixin for all prediction trainers to implement the predict method.
    """

    @abstractmethod
    def predict(self, docs: Dict[int, str]) -> Dict[int, PredictionResult]:
        ...
