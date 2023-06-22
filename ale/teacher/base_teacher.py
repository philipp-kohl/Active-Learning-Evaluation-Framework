from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ale.corpus.corpus import Corpus
from ale.trainer.base_trainer import Predictor


class BaseTeacher(ABC):
    """
    abstract teacher class
    """

    def __init__(
        self, corpus: Corpus, predictor: Predictor, labels: List[Any], seed: int
    ):
        self.labels = labels
        self.corpus = corpus
        self.predictor = predictor
        self.seed = seed

    @abstractmethod
    def propose(self, potential_ids: List[int], actual_step_size: int, actual_budget: int) -> List[int]:
        """
        :type potential_ids: object
        :type actual_step_size: int
        :return: data_uri and List of data indices
        """
        pass

    def after_train(self, metrics: Dict):
        """
        Will be called after every training, except the initial train

        :type metrics: object
        :return:
        """

    def after_initial_train(self, metrics: Dict):
        """
        Will be called after the initial train

        :type metrics: object
        :return:
        """
