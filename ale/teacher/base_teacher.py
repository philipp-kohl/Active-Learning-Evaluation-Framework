from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple

from ale.corpus.corpus import Corpus
from ale.trainer.base_trainer import Predictor
from ale.config import NLPTask
from ale.trainer.prediction_result import PredictionResult


class BaseTeacher(ABC):
    """
    abstract teacher class
    """

    def __init__(
        self, corpus: Corpus, predictor: Predictor, labels: List[Any], seed: int, nlp_task: NLPTask
    ):
        self.labels = labels
        self.corpus = corpus
        self.predictor = predictor
        self.seed = seed
        self.nlp_task = nlp_task

        self.compute_function: Callable[
            [List[PredictionResult], int], Tuple[Dict[str, float], Dict[str, float]]] = {
            NLPTask.CLS: self.compute_cls,
            NLPTask.NER: self.compute_ner
        }[nlp_task]

    @abstractmethod
    def propose(self, potential_ids: List[int], actual_step_size: int, actual_budget: int) -> List[int]:
        """
        :type potential_ids: object
        :type actual_step_size: int
        :return: data_uri and List of data indices
        """
        pass
    
    @abstractmethod
    def compute_cls(self, predictions: List[PredictionResult], step_size: int) -> List[int]:
        """
        Computes the order in which the samples are proposed according to the teacher used.
        Args:
            - predictions (List[PredictionResult]): prediction results of the samples to be proposed
        Returns:
            - List[int]: ordered list of indices of the documents
        """
        pass

    @abstractmethod
    def compute_ner(self, predictions: List[PredictionResult], step_size: int) -> List[int]:
        """
        Computes the order in which the samples are proposed according to the teacher used.
        Args:
            - predictions (List[PredictionResult]): prediction results of the samples to be proposed
        Returns:
            - List[int]: ordered list of indices of the documents
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
