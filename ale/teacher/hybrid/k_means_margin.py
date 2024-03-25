from abc import ABC
from typing import List, Dict, Optional
import numpy as np
import random

from ale.config import NLPTask
from ale.registry.registerable_teacher import TeacherRegistry
from ale.teacher.base_teacher import BaseTeacher
from ale.corpus.corpus import Corpus
from ale.teacher.exploitation.aggregation_methods import AggregationMethod
from ale.trainer.base_trainer import Predictor
from ale.teacher.exploitation.margin_confidence import MarginTeacher
from ale.teacher.exploration.k_means import KMeansTeacher

@TeacherRegistry.register("k-means-margin-confidence")
class KMeansMarginTeacher(BaseTeacher, ABC):
    """
    KMeans margin teacher: chooses nearest neighbors and then selects instances with lowest margin
    """
    def __init__(
        self,
        corpus: Corpus,
        predictor: Predictor,
        seed: int,
        labels: List[any],
        nlp_task: NLPTask,
        aggregation_method: Optional[AggregationMethod]
    ):
        super().__init__(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task,
            aggregation_method=aggregation_method
        )
        self.margin_teacher = MarginTeacher(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task,
            aggregation_method=aggregation_method
            )
        
        self.kmeans_teacher = KMeansTeacher(
            corpus=corpus,
            predictor=predictor,
            seed=seed,
            labels=labels,
            nlp_task=nlp_task
        )

    def propose(self, potential_ids: List[int], step_size: int, budget: int) -> List[int]:        
        """
        Use KMeans with step_size*5 and then select with margin
        """
        kmeans_results: List[int] = self.kmeans_teacher.propose(potential_ids,step_size*5, budget) # TODO magic number as config parameter?
        out_ids: List[int] = self.margin_teacher.propose(kmeans_results, step_size, budget)

        return out_ids