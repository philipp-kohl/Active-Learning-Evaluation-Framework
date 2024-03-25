from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Tuple, Optional

from ale.corpus.corpus import Corpus
from ale.teacher.exploitation.aggregation_methods import AggregationMethod, Aggregation
from ale.trainer.base_trainer import Predictor
from ale.config import NLPTask
from ale.trainer.prediction_result import PredictionResult


class BaseTeacher(ABC):
    """
    abstract teacher class
    """

    def __init__(
            self, corpus: Corpus, predictor: Predictor, labels: List[Any], seed: int, nlp_task: NLPTask,
            aggregation_method: Optional[AggregationMethod]
    ):
        self.labels = labels
        self.corpus = corpus
        self.predictor = predictor
        self.seed = seed
        self.nlp_task = nlp_task

        if aggregation_method is not None:  # For exploitation based approaches in Entity Recognition
            self.aggregation_method = aggregation_method
            self.aggregate_function = Aggregation(self.aggregation_method).get_aggregate_function()

        self.compute_function: Callable[
            [Dict[int, PredictionResult], int], List[int]] = {
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
    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> List[int]:
        """
        Computes the order in which the samples are proposed according to the teacher used.
        Args:
            - predictions (Dict[int,PredictionResult]): key: id of doc, value: prediction result of doc
        Returns:
            - List[int]: ordered list of indices of the documents
        """
        pass

    @abstractmethod
    def compute_ner(self, predictions: Dict[int, PredictionResult], step_size: int) -> List[int]:
        """
        Computes the order in which the samples are proposed according to the teacher used.
        Args:
            - predictions (Dict[int,PredictionResult]): key: id of doc, value: prediction result of doc
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
