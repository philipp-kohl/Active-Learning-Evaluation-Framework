import logging
import os
import pickle
import tempfile
from typing import List, Dict, Any, Callable, Optional

from mlflow.entities import Run

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.mlflowutils import mlflow_utils
from ale.teacher.exploitation.aggregation_methods import AggregationMethod, Aggregation
from ale.trainer.prediction_result import PredictionResult
from ale.trainer.predictor import Predictor

logger = logging.getLogger(__name__)

class BaseTeacher:
    """
    abstract teacher class
    """

    def __init__(
            self, corpus: Corpus, predictor: Predictor, labels: List[Any], seed: int, nlp_task: NLPTask,
            aggregation_method: Optional[AggregationMethod] = None
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

    def propose(self, potential_ids: List[int], actual_step_size: int, actual_budget: int) -> List[int]:
        """
        :type potential_ids: object
        :type actual_step_size: int
        :return: List of data indices
        """
        pass

    def compute_cls(self, predictions: Dict[int, PredictionResult], step_size: int) -> List[int]:
        """
        Computes the order in which the samples are proposed according to the teacher used.
        Args:
            - predictions (Dict[int,PredictionResult]): key: id of doc, value: prediction result of doc
        Returns:
            - List[int]: ordered list of indices of the documents
        """
        pass

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

    def store_state(self, run: Run):
        pass

    def restore_from_artifacts(self, run: Run):
        pass

    def store_state_objects(self, run: Run, objects: Dict[str, Any]):
        logger.info("Store state for teacher resume")
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, obj in objects.items():
                temp_filepath = os.path.join(temp_dir, name)
                logger.info(f"Store teacher state to: {temp_filepath}")
                with open(temp_filepath, "wb") as f:
                    pickle.dump(obj, f)

                mlflow_utils.log_artifact(run, temp_filepath, "teacher_state")
            logger.info("Teacher state stored successfully")

    def restore_state_objects(self, run: Run, names: List[str]) -> Dict[str, Any]:
        logger.info("Restore state for teacher resume")

        state: Dict[str, Any] = {}
        for name in names:
            artifact_path = f"teacher_state/{name}"
            logger.info(f"Restore state from: {run.info.run_id}/{artifact_path}")
            model_path = mlflow_utils.load_artifact(run, artifact_path)
            with open(model_path, 'rb') as temp_file:
                state[name] = pickle.load(temp_file)
        logger.info(f"Restore state successfully")

        return state
