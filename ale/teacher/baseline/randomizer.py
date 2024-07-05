import random
from abc import ABC
from typing import List, Any, Dict

from ale.config import NLPTask
from ale.corpus.corpus import Corpus
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.trainer.predictor import Predictor

@TeacherRegistry.register("randomizer")
class RandomTeacher(BaseTeacher, ABC):
    """
    Random teacher: folds the data initially and makes random choices from iterated folds as propose is called
    """

    def __init__(
        self,
        corpus: Corpus,
        predictor: Predictor,
        seed: int,
        labels: List[Any],
        nlp_task: NLPTask
    ):
        super().__init__(
            corpus=corpus, predictor=predictor, seed=seed, labels=labels, nlp_task=nlp_task
        )
        random.seed(self.seed)

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:
        next_batch = random.sample(potential_ids, step_size)
        return next_batch
